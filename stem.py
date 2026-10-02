#!/usr/bin/env python3
"""词形归一（lemmatization / stemming）引擎。

单进程、单线程、纯标准库。流程：
载入并校验（词表、规则、例外）-> 逐词归一 -> 建反查索引 -> 召回 / 渲染页面。

口径见仓库 README：例外优先于规则；规则按后缀长度降序、同长按序号升序，
第一条条件全满足的规则生效；派生词不合并。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

VOWELS = frozenset("aeiou")
SIBILANT_ENDS = ("ch", "sh", "ss")
POS_TAGS = frozenset({"n", "v", "adj", "adv", "quant", "other"})
EXCEPTION_TYPES = frozenset({"不规则", "保护", "歧义", "数量词"})

WORD_MIN_LEN = 1
WORD_MAX_LEN = 64


class InputError(Exception):
    """输入非法：消息里必须含文件与行号。"""


def is_valid_word(token: str) -> bool:
    return (
        WORD_MIN_LEN <= len(token) <= WORD_MAX_LEN
        and all("a" <= ch <= "z" for ch in token)
    )


def preprocess(token: str) -> str:
    """去首尾空白 + 转小写。合法性由调用方校验。"""
    return token.strip().lower()


def _iter_records(path: str, min_fields: int, max_fields: int):
    """逐行读出非注释、非空 TSV 记录；产出 (物理行号, 字段列表)。"""
    try:
        raw = open(path, "r", encoding="utf-8").read()
    except OSError as exc:
        raise InputError(f"{path}: 无法读取: {exc}") from None
    if raw.startswith("\ufeff"):
        raise InputError(f"{path}:1: 文件不得带 BOM")
    if "\r" in raw:
        raise InputError(f"{path}: 不允许 CR 换行（须 LF）")
    for lineno, line in enumerate(raw.split("\n"), start=1):
        if line == "" and lineno == raw.count("\n") + 1:
            # split 末尾的空串不是真实行；真实的空行在上面已经被计数并忽略。
            break
        if line == "":
            continue
        if line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if not (min_fields <= len(fields) <= max_fields):
            raise InputError(
                f"{path}:{lineno}: 字段数非法（应为 {min_fields}"
                + (f"~{max_fields}" if max_fields != min_fields else "")
                + f" 列，实际 {len(fields)} 列）"
            )
        yield lineno, fields


def load_words(path: str) -> list[tuple[str, frozenset[str]]]:
    """读词表，保持出现次序，重复词形按首次计。返回 [(词形, 词性集合)]。"""
    words: list[tuple[str, frozenset[str]]] = []
    seen: set[str] = set()
    for lineno, fields in _iter_records(path, 2, 3):
        word = preprocess(fields[0])
        if not is_valid_word(word):
            raise InputError(f"{path}:{lineno}: 非法词形 {fields[0]!r}")
        tags_text = fields[1].strip()
        if not tags_text:
            raise InputError(f"{path}:{lineno}: 词性为空")
        tags = frozenset(tags_text.split(","))
        if not tags or not tags <= POS_TAGS:
            raise InputError(f"{path}:{lineno}: 非法词性 {tags_text!r}")
        if word in seen:
            continue
        seen.add(word)
        words.append((word, tags))
    return words


def load_rules(path: str) -> list[tuple[str, str, int, str | None, bool]]:
    """规则表 -> 有序规则列表（后缀长度降序、同长按序号升序）。

    每条规则为 (后缀, 替换串或''表示删除, 最小词干, 要求, 是否短化)。
    """
    rules: list[tuple[int, int, str, str, int, str | None, bool]] = []
    for lineno, fields in _iter_records(path, 6, 6):
        seq_text, suffix, repl_text, min_stem_text, req_text, fix_text = fields
        suffix = suffix.strip()
        if not suffix or not all("a" <= ch <= "z" for ch in suffix):
            raise InputError(f"{path}:{lineno}: 非法后缀 {suffix!r}")
        if repl_text == "-":
            replacement = ""
        else:
            replacement = repl_text.strip()
            if not replacement or not all("a" <= ch <= "z" for ch in replacement):
                raise InputError(f"{path}:{lineno}: 非法替换 {repl_text!r}")
        if not seq_text.strip().isdigit():
            raise InputError(f"{path}:{lineno}: 序号须为非负整数")
        seq = int(seq_text)
        min_stem = min_stem_text.strip()
        if not min_stem.isdigit() or int(min_stem) < 0:
            raise InputError(f"{path}:{lineno}: 最小词干须为非负整数")
        req = req_text.strip()
        if req == "-":
            requirement = None
        elif req in ("形容词", "咝音"):
            requirement = req
        else:
            raise InputError(f"{path}:{lineno}: 非法要求 {req!r}")
        fix = fix_text.strip()
        if fix == "短化":
            shorten = True
        elif fix == "-":
            shorten = False
        else:
            raise InputError(f"{path}:{lineno}: 非法修补 {fix!r}")
        rules.append((len(suffix), seq, suffix, replacement, int(min_stem),
                      requirement, shorten))
    rules.sort(key=lambda r: (-r[0], r[1]))
    return [(suffix, repl, min_stem, req, shorten)
            for _, _, suffix, repl, min_stem, req, shorten in rules]


def load_exceptions(path: str) -> dict[str, str]:
    """例外表 词形 -> 归一形式；同一词形两个归一形式即输入错误。"""
    exceptions: dict[str, str] = {}
    for lineno, fields in _iter_records(path, 3, 3):
        word = preprocess(fields[0])
        if not is_valid_word(word):
            raise InputError(f"{path}:{lineno}: 非法词形 {fields[0]!r}")
        target = preprocess(fields[1])
        if not is_valid_word(target):
            raise InputError(f"{path}:{lineno}: 非法归一形式 {fields[1]!r}")
        kind = fields[2].strip()
        if kind not in EXCEPTION_TYPES:
            raise InputError(f"{path}:{lineno}: 非法例外类型 {kind!r}")
        if word in exceptions and exceptions[word] != target:
            raise InputError(
                f"{path}:{lineno}: 词形 {word} 出现两个归一形式"
            )
        exceptions[word] = target
    return exceptions


def load_families(path: str) -> dict[str, str]:
    """人工标注 词形 -> 族标签。"""
    families: dict[str, str] = {}
    for lineno, fields in _iter_records(path, 3, 3):
        word = preprocess(fields[0])
        if not is_valid_word(word):
            raise InputError(f"{path}:{lineno}: 非法词形 {fields[0]!r}")
        label = preprocess(fields[1])
        if not is_valid_word(label):
            raise InputError(f"{path}:{lineno}: 非法族标签 {fields[1]!r}")
        if word in families and families[word] != label:
            raise InputError(
                f"{path}:{lineno}: 词形 {word} 出现两个族标签"
            )
        families[word] = label
    return families


def _ends_sibilant(stem: str) -> bool:
    return stem.endswith(("x", "z", "ch", "sh", "ss"))


def _is_cvc(stem: str) -> bool:
    """末三字符是否为 辅音+元音+辅音，且末位不是 w/x/y。"""
    if len(stem) < 3:
        return False
    c1, v, c2 = stem[-3], stem[-2], stem[-1]
    return (
        c1 not in VOWELS
        and v in VOWELS
        and c2 not in VOWELS
        and c2 not in "wxy"
    )


def _count_vowels(token: str) -> int:
    return sum(1 for ch in token if ch in VOWELS)


class Stemmer:
    """规则 + 例外的归一器；词表用于「要求」与补 e 查表。"""

    def __init__(self, words, rules, exceptions):
        self.pos: dict[str, frozenset[str]] = dict(words)
        self.rules = rules
        self.exceptions = exceptions

    def normalize(self, word: str) -> str:
        if word in self.exceptions:
            return self.exceptions[word]
        for suffix, replacement, min_stem, requirement, shorten in self.rules:
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if len(stem) < min_stem:
                continue
            candidate = stem + replacement
            if shorten:
                candidate = self._repair(candidate)
            if requirement == "咝音" and not _ends_sibilant(stem):
                continue
            if requirement == "形容词":
                tags = self.pos.get(candidate)
                if tags is None or not (tags & {"adj", "adv"}):
                    continue
            return candidate
        return word

    def _repair(self, candidate: str) -> str:
        # ① 末尾两个相同辅音（非 l/s/z）且长度 >= 4：去双写。
        if (
            len(candidate) >= 4
            and candidate[-1] == candidate[-2]
            and candidate[-1] not in "lsz"
        ):
            return candidate[:-1]
        # ② 补 e 后在词表内。
        if candidate + "e" in self.pos:
            return candidate + "e"
        # ③ 单元音 + 末三 CVC + 末位非 w/x/y：补 e。
        if _count_vowels(candidate) == 1 and _is_cvc(candidate):
            return candidate + "e"
        # ④ 原样。
        return candidate


def build_stemmer(words, rules_path: str | None = None,
                  exceptions_path: str | None = None) -> Stemmer:
    here = os.path.dirname(os.path.abspath(__file__))
    if rules_path is None:
        rules_path = os.path.join(here, "data", "rules.tsv")
    if exceptions_path is None:
        exceptions_path = os.path.join(here, "data", "exceptions.tsv")
    rules = load_rules(rules_path)
    exceptions = load_exceptions(exceptions_path)
    return Stemmer(words, rules, exceptions)


def build_model(words, stemmer: Stemmer):
    """逐词归一并建反查索引。返回 (归一结果按词表序, 反查索引有序)。"""
    stems: list[str] = [stemmer.normalize(w) for w, _ in words]
    index: dict[str, list[str]] = {}
    for (word, _), canonical in zip(words, stems):
        index.setdefault(canonical, []).append(word)
    ordered = {canonical: sorted(members)
               for canonical, members in sorted(index.items())}
    return stems, ordered


def evaluate(index: dict[str, list[str]], labels: dict[str, str]) -> dict:
    """用人工标注统计错误合并（同族内出现多个族标签）。

    返回 {bad_families: set, bad_pairs, total_pairs, ratio}。
    total_pairs 为落在同一引擎簇里的已标注词对数（含正确合并）。
    """
    bad_families: set[str] = set()
    bad_pairs = 0
    total_pairs = 0
    for canonical, members in index.items():
        tagged = [labels[w] for w in members if w in labels]
        if len(set(tagged)) > 1:
            bad_families.add(canonical)
        n = len(tagged)
        total_pairs += n * (n - 1) // 2
        counts: dict[str, int] = {}
        for label in tagged:
            counts[label] = counts.get(label, 0) + 1
        same = sum(c * (c - 1) // 2 for c in counts.values())
        bad_pairs += n * (n - 1) // 2 - same
    ratio = (bad_pairs / total_pairs) if total_pairs else 0.0
    return {
        "bad_families": bad_families,
        "bad_pairs": bad_pairs,
        "total_pairs": total_pairs,
        "ratio": ratio,
    }


def render_run(words, stems, ordered, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stem_path = os.path.join(out_dir, "stem.tsv")
    index_path = os.path.join(out_dir, "index.tsv")
    with open(stem_path, "w", encoding="utf-8", newline="") as fh:
        for (word, _), canonical in zip(words, stems):
            fh.write(f"{word}\t{canonical}\n")
    with open(index_path, "w", encoding="utf-8", newline="") as fh:
        for canonical, members in ordered.items():
            fh.write(f"{canonical}\t{','.join(members)}\n")


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #

def page_data(words, stems, ordered, evaluation) -> dict:
    families = []
    for canonical, members in ordered.items():
        families.append({
            "stem": canonical,
            "members": members,
            "bad": canonical in evaluation["bad_families"],
        })
    return {
        "words": [w for w, _ in words],
        "families": families,
        "badPairs": evaluation["bad_pairs"],
        "totalPairs": evaluation["total_pairs"],
        "ratio": round(evaluation["ratio"], 6),
    }


def render_page(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "web", "template.html"
    )
    with open(template_path, "r", encoding="utf-8") as fh:
        template = fh.read()
    return template.replace("/*__DATA__*/null", payload, 1)


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def cmd_run(args) -> int:
    words = load_words(args.words)
    stemmer = build_stemmer(words)
    stems, ordered = build_model(words, stemmer)
    render_run(words, stems, ordered, args.out)
    print(f"words={len(words)} families={len(ordered)}")
    return 0


def cmd_recall(args) -> int:
    words = load_words(args.words)
    stemmer = build_stemmer(words)
    _, ordered = build_model(words, stemmer)
    try:
        raw = open(args.queries, "r", encoding="utf-8").read()
    except OSError:
        raise InputError(f"{args.queries}: 无法读取") from None
    if raw.startswith("\ufeff") or "\r" in raw:
        raise InputError(f"{args.queries}: 编码或换行非法")
    out = []
    for lineno, line in enumerate(raw.split("\n"), start=1):
        if raw.endswith("\n") and lineno == raw.count("\n") + 1:
            break
        if line == "" or line.lstrip().startswith("#"):
            continue
        query = preprocess(line)
        if not is_valid_word(query):
            raise InputError(f"{args.queries}:{lineno}: 非法查询词 {line!r}")
        members = ordered.get(stemmer.normalize(query), [])
        out.append(f"{query}\t{','.join(members)}")
    sys.stdout.write("\n".join(out))
    if out:
        sys.stdout.write("\n")
    return 0


def cmd_web(args) -> int:
    words = load_words(args.words)
    stemmer = build_stemmer(words)
    stems, ordered = build_model(words, stemmer)
    if args.families:
        labels = load_families(args.families)
    else:
        labels = {}
    evaluation = evaluate(ordered, labels)
    data = page_data(words, stems, ordered, evaluation)
    html = render_page(data)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write(html)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stem.py", description="词形归一")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="归一并建反查索引")
    p_run.add_argument("--words", required=True)
    p_run.add_argument("--out", default="var")
    p_run.set_defaults(func=cmd_run)

    p_recall = sub.add_parser("recall", help="按查询清单召回")
    p_recall.add_argument("--words", required=True)
    p_recall.add_argument("--queries", required=True)
    p_recall.set_defaults(func=cmd_recall)

    p_web = sub.add_parser("web", help="生成连线图页面")
    p_web.add_argument("--words", required=True)
    p_web.add_argument("--families", default=None,
                       help="人工标注（families.tsv），用于标红错误合并")
    p_web.add_argument("--out", default="web/index.html")
    p_web.set_defaults(func=cmd_web)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
