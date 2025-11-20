import json
import sys
from .config import PipelineConfig
from .pipeline import Pipeline


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    cfg = PipelineConfig()
    pipe = Pipeline(cfg)
    result = pipe.run(text)

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
