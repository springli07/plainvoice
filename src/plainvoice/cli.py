"""Command line interface; no hidden network calls for diagnostic commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from plainvoice import __version__
from plainvoice.analysis import analyze
from plainvoice.profiles import default_profile, load_profile
from plainvoice.providers import RewriteError, rewrite, validate_text


def _read(path: str) -> str:
    if path == "-":
        text = sys.stdin.read(50_001)
    else:
        target = Path(path)
        if target.stat().st_size > 256 * 1024:
            raise ValueError("输入文件超过 256 KiB。")
        text = target.read_text(encoding="utf-8")
    return validate_text(text)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _write_json(path: str, value: Any) -> None:
    # Creation only: a typo must not overwrite an input or an existing profile.
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")


def _load_json(path: str) -> dict[str, Any]:
    target = Path(path)
    if target.stat().st_size > 4_000_000:
        raise ValueError("配置文件过大。")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("配置必须为 JSON 对象。")
    return data


def _tokenizer(name: str) -> Any:
    try:
        import tiktoken
    except ImportError as exc:
        raise ValueError("该命令需要安装 plainvoice[tokens]。") from exc
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:
        raise ValueError("Tokenizer 加载失败；请检查名称、缓存及所需下载域名。") from exc


def _corpus(directory: str) -> list[str]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError("语料路径必须为目录。")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".txt", ".md"} and p.is_file())
    if not paths or len(paths) > 10_000:
        raise ValueError("语料目录需要 1 至 10000 个 .txt 或 .md 文件。")
    total = 0
    docs = []
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("语料不能包含指向目录之外的符号链接。")
        size = path.stat().st_size
        total += size
        if size > 4_000_000 or total > 50_000_000:
            raise ValueError("单篇语料最多 4 MB，合计最多 50 MB。")
        docs.append(path.read_text(encoding="utf-8"))
    return docs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plainvoice", description="中文写作偏好诊断与可检查的 token 文风控制。"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("analyze", "离线检查写作偏好"), ("rewrite", "通过模型最小改写")):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("file", nargs="?", default="-", help="UTF-8 文件，- 表示标准输入")
        item.add_argument("--profile")
        item.add_argument("--json", action="store_true")
        if name == "rewrite":
            item.add_argument("--backend", choices=("openai", "local"),
                              default=os.environ.get("PLAINVOICE_BACKEND", "openai"))
            item.add_argument("--model")
            item.add_argument("--base-url")
            item.add_argument("--api-key-env", default="PLAINVOICE_API_KEY")
            item.add_argument("--bias-table")
            item.add_argument("--tokenizer-id")
            item.add_argument("--allow-static-bias", action="store_true")
            item.add_argument("--allow-positive", action="store_true")
            item.add_argument("--timeout", type=float, default=60)
    serve = sub.add_parser("serve", help="启动仅本机访问的中文工作台")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    compile_cmd = sub.add_parser("compile", help="编译本地短语解码计划")
    compile_cmd.add_argument("--profile")
    compile_cmd.add_argument("--tokenizer", default="cl100k_base")
    compile_cmd.add_argument("--allow-positive", action="store_true")
    compile_cmd.add_argument("--out", required=True)
    fit = sub.add_parser("fit", help="从自己的对照语料估计有界 token 偏好")
    fit.add_argument("--human", required=True)
    fit.add_argument("--model-corpus", required=True)
    fit.add_argument("--tokenizer", default="cl100k_base")
    fit.add_argument("--min-count", type=int, default=5)
    fit.add_argument("--max-bias", type=float, default=1)
    fit.add_argument("--allow-positive", action="store_true")
    fit.add_argument("--out", required=True)
    demo = sub.add_parser("demo", help="运行无需模型的机制演示，不生成文章")
    demo.add_argument("--json", action="store_true")
    return parser


def demo() -> dict[str, Any]:
    from plainvoice.decoding import PhraseBiasPlan, TokenPhrase

    plan = PhraseBiasPlan("illustrative-tokenizer", 2.0, (TokenPhrase((1, 2, 3), -1.0),), 8)
    return {
        "demo": "mechanism-only",
        "description": "使用示意 token ID 展示偏置触发条件，没有调用或模拟语言模型。",
        "analysis": analyze("不难看出，这个版本仍需测试。我们计划明天发布。"),
        "phrase_token_ids": [1, 2, 3],
        "without_prefix": plan.biases([7]),
        "with_prefix": plan.biases([1, 2]),
        "limitations": "机制正确不代表文风更好；需在独立语料上盲评并检查事实。",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "serve":
            from plainvoice.server import create_server

            server = create_server(args.host, args.port)
            print(f"PlainVoice 已启动，监听本机端口 {server.server_port}。按 Ctrl+C 停止。", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0
        if args.command == "demo":
            print(_json(demo()))
            return 0
        profile = load_profile(args.profile) if getattr(args, "profile", None) else default_profile()
        if args.command == "analyze":
            result = analyze(_read(args.file), profile)
            if args.json:
                print(_json(result))
            else:
                print(f"{result['character_count']} 字符，{len(result['findings'])} 处可复核的偏好命中。")
                for hit in result["findings"]:
                    print(f"[{hit['start']}:{hit['end']}] {hit['text']}：{hit['message']}")
                print(result["disclaimer"])
        elif args.command == "rewrite":
            result = rewrite(
                _read(args.file), profile,
                backend=args.backend, model=args.model, base_url=args.base_url,
                api_key=os.environ.get(args.api_key_env), timeout=args.timeout,
                token_bias=_load_json(args.bias_table) if args.bias_table else None,
                tokenizer_id=args.tokenizer_id, allow_static_bias=args.allow_static_bias,
                allow_positive=args.allow_positive,
            )
            print(_json(result) if args.json else result["text"])
        elif args.command == "compile":
            from plainvoice.decoding import compile_plan

            plan = compile_plan(profile, _tokenizer(args.tokenizer), args.tokenizer,
                                allow_positive=args.allow_positive)
            _write_json(args.out, plan.to_dict())
            print(f"已写入 {len(plan.phrases)} 条短语编码；仅用于支持逐步 logits 的本地后端。")
        elif args.command == "fit":
            from plainvoice.corpus import fit_token_bias

            table = fit_token_bias(
                _corpus(args.human), _corpus(args.model_corpus), _tokenizer(args.tokenizer),
                args.tokenizer, min_count=args.min_count, max_bias=args.max_bias,
                allow_positive=args.allow_positive,
            )
            _write_json(args.out, table)
            print(f"已写入 {len(table['biases'])} 项语料偏好；不代表作者身份判断。")
        return 0
    except (ValueError, OSError, RewriteError) as exc:
        print(f"plainvoice: {exc}", file=sys.stderr)
        return 2
