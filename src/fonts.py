from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re

from .embeddings import IMAGE_MODEL, embed_images


PHRASES = [
    "Beyond the ordinary",
    "A curious collection",
    "Patterns in plain sight",
    "Something unexpected",
    "Quietly extraordinary",
]
FAMILIES = """roboto opensans lato montserrat raleway nunito ubuntu worksans inter
oswald barlow anton bebasneue archivo sourcesans3 dmsans manrope karla rubik
merriweather playfairdisplay lora cormorantgaramond ebgaramond librebaskerville
crimsontext bitter abrilfatface dmserifdisplay bodonimoda cinzel prata vollkorn
spectral literata alice cardo oldstandardtt yesevaone
pacifico lobster dancingscript greatvibes sacramento allura parisienne
alexbrush tangerine pinyonscript italianno petitformalscript mrdafoe yellowtail
satisfy kaushanscript courgette marckscript cookie norican
monoton bungee bungeeshade bungeeinline ewert fasterone vastshadow
frederickathegreat rye unifrakturmaguntia unifrakturcook nosifer creepster
rubikmoonrocks rubikmicrobe rubikglitch frijolesspecialelite
specialelite pressstart2p blackopsone wallpoet limelight fascinate
permanentmarker caveat indieflower amatic sc patrickhand shadowsintolight
architectsdaughter rockSalt gloriahallelujah""".lower().split()
FAMILIES = [name for name in FAMILIES if name not in {"sc", "frijolesspecialelite", "amatic"}]
FAMILIES += ["amaticsc", "frijole"]


def fetch_font(slug, folder):
    target = folder / "fonts" / slug
    cached = target / "source.json"
    if cached.exists():
        return json.loads(cached.read_text())

    import requests

    base = f"https://raw.githubusercontent.com/google/fonts/main/ofl/{slug}"
    response = requests.get(f"{base}/METADATA.pb", timeout=30)
    if response.status_code != 200:
        return {"slug": slug, "error": f"metadata HTTP {response.status_code}"}
    metadata = response.text
    blocks = re.findall(r"fonts \{(.*?)\n\}", metadata, re.S)
    regular = next((b for b in blocks if 'style: "normal"' in b and "weight: 400" in b), None)
    regular = regular or next((b for b in blocks if 'style: "normal"' in b), "")
    match = re.search(r'filename: "([^"]+)"', regular)
    if not match:
        return {"slug": slug, "error": "no upright font file"}
    filename = match.group(1)
    font = requests.get(f"{base}/{requests.utils.quote(filename)}", timeout=60)
    font.raise_for_status()
    license = requests.get(f"{base}/OFL.txt", timeout=30)
    license.raise_for_status()
    target.mkdir(parents=True, exist_ok=True)
    (target / filename).write_bytes(font.content)
    (target / "OFL.txt").write_text(license.text)
    (target / "METADATA.pb").write_text(metadata)
    info = {
        "slug": slug,
        "name": re.search(r'^name: "([^"]+)"', metadata, re.M).group(1),
        "categories": re.findall(r'^category: "([^"]+)"', metadata, re.M),
        "filename": filename,
        "source": f"https://github.com/google/fonts/tree/main/ofl/{slug}",
        "sha256": hashlib.sha256(font.content).hexdigest(),
    }
    cached.write_text(json.dumps(info, indent=2) + "\n")
    return info


def render(info, phrase, folder):
    from PIL import Image, ImageDraw, ImageFont
    from fontTools.ttLib import TTFont

    path = folder / "fonts" / info["slug"] / info["filename"]
    with TTFont(path) as metadata:
        axes = metadata["fvar"].axes if "fvar" in metadata else []

    def at_size(size):
        font = ImageFont.truetype(str(path), size)
        if axes:
            values = [
                min(a.maxValue, max(a.minValue, 400 if a.axisTag == "wght" else a.defaultValue))
                for a in axes
            ]
            font.set_variation_by_axes(values)
        return font

    image = Image.new("RGB", (640, 256), "white")
    draw = ImageDraw.Draw(image)
    font = at_size(54)
    bounds = draw.textbbox((0, 0), phrase, font=font)
    if bounds[2] - bounds[0] > 590:
        font = at_size(int(54 * 590 / (bounds[2] - bounds[0])))
        bounds = draw.textbbox((0, 0), phrase, font=font)
    x = (640 - bounds[2] + bounds[0]) / 2 - bounds[0]
    y = (256 - bounds[3] + bounds[1]) / 2 - bounds[1]
    draw.text((x, y), phrase, fill="#191b24", font=font)
    return image


def prepare(folder):
    for path in (folder, folder / "fonts", folder / "images"):
        path.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "font-metadata.json"
    if metadata_path.exists():
        fonts = json.loads(metadata_path.read_text())["fonts"]
    elif (folder / "results.json").exists():
        # retain the original family order when using the existing dataset
        previous = json.loads((folder / "results.json").read_text())
        fonts = [fetch_font(font["slug"], folder) for font in previous["fonts"]]
    else:
        families = list(dict.fromkeys(FAMILIES))
        with ThreadPoolExecutor(max_workers=8) as pool:
            fonts = list(pool.map(lambda slug: fetch_font(slug, folder), families))
    errors = [font for font in fonts if "error" in font]
    fonts = [font for font in fonts if "error" not in font]
    if len(fonts) < 2:
        raise ValueError("Need at least two font families")
    for font in fonts:
        for i, phrase in enumerate(PHRASES):
            target = folder / "images" / f"{font['slug']}-{i}.png"
            if not target.exists():
                render(font, phrase, folder).save(target)
    result = {"fonts": fonts, "phrases": PHRASES, "skipped": errors}
    metadata_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def embed(folder, metadata, device=None):
    fonts = metadata["fonts"]
    identity = hashlib.sha256(
        json.dumps([IMAGE_MODEL, PHRASES, fonts, "render-v1"]).encode()
    ).hexdigest()
    files = [
        folder / "images" / f"{font['slug']}-{i}.png" for font in fonts for i in range(len(PHRASES))
    ]
    return embed_images(
        files, folder / "embeddings.npz", identity, (len(fonts), len(PHRASES)), device
    )
