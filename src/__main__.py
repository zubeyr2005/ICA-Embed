import argparse
from pathlib import Path

from . import DATA, RESULTS
from . import combinations, data, embeddings, fonts, pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Prepare embeddings and export ICA component scores"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for kind in ("text", "fonts"):
        command = commands.add_parser(kind)
        command.add_argument("stage", choices=("prepare", "embed", "fit"))
        command.add_argument("--data", type=Path, default=DATA / kind)
        command.add_argument("--output", type=Path, default=RESULTS / kind)
        command.add_argument("--components", type=int, default=32 if kind == "text" else 16)
        if kind == "text":
            command.add_argument("--source", type=Path)
            command.add_argument("--max-parts", type=int)
        else:
            command.add_argument("--device")
    combine = commands.add_parser("combine")
    combine.add_argument("--results", type=Path, default=RESULTS / "text")
    combine.add_argument("--examples", type=Path, default=DATA / "text" / "examples.json")
    combine.add_argument("--cache", type=Path, default=DATA / "text" / "example-embeddings.npz")
    combine.add_argument("--preset", default="all")
    args = parser.parse_args()

    if args.command == "combine":
        combinations.run(args.results, args.examples, args.cache, args.preset)
    elif args.command == "text":
        if args.stage == "prepare":
            if args.source is None:
                parser.error("text prepare needs --source")
            data.prepare(args.source, args.data)
        elif args.stage == "embed":
            embeddings.embed_text(args.data, args.max_parts)
        else:
            pipeline.fit_text(args.data, args.output, args.components)
    elif args.stage == "prepare":
        fonts.prepare(args.data)
    elif args.stage == "embed":
        metadata = fonts.prepare(args.data)
        fonts.embed(args.data, metadata, args.device)
    else:
        pipeline.fit_fonts(args.data, args.output, args.components)


if __name__ == "__main__":
    main()
