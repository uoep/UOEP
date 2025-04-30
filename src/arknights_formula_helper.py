# arknights_formula_helper.py

from enum import Enum
from typing import Sequence, Callable, Iterable, Literal, Optional, Union
import sys
import os
import re
import json
import csv
import traceback
import argparse
import time


_OPERATOR_INFO_FILENAME = "arknights-operator-info.csv"
_INPUT_FILENAME = "input.txt"
_OUTPUT_FILENAME = "output.txt"
_DEBUG = False


def main() -> None:
    args = parse_args()
    if args.file:
        run_file(args.transform_mode, args.minify_policy, args.dry_run)
    else:
        run_repl(args.transform_mode, args.minify_policy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Effortlessly tailor your Arknights formulas to "
                    "various battlefields with all possible buffs",
    )
    parser.add_argument(
        "--from",
        choices=["damage", "endurance"],
        default="damage",
        dest="from_type",
        help="formula source type",
    )
    parser.add_argument(
        "--to",
        choices=["buff", "decay", "enemy"],
        default="buff",
        dest="to_type",
        help="formula target type",
    )
    parser.add_argument(
        "--minify",
        choices=["auto", "off", "max"],
        help="whether to minify the generated formula",
    )
    parser.add_argument(
        "--file",
        action="store_true",
        help=f"read input from `{_INPUT_FILENAME}` and "
             f"write output to `{_OUTPUT_FILENAME}`",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="do not write the disk"
    )
    args = parser.parse_args()
    if args.from_type == "damage":
        if args.to_type == "buff":
            args.transform_mode = TransformMode.DAMAGE_BUFF
        elif args.to_type == "decay":
            args.transform_mode = TransformMode.DAMAGE_DECAY
        elif args.to_type == "enemy":
            args.transform_mode = TransformMode.DAMAGE_ENEMY
        else:
            Assert.never()
    elif args.from_type == "endurance":
        if args.to_type == "buff":
            args.transform_mode = TransformMode.ENDURANCE_BUFF
        elif args.to_type == "decay":
            args.transform_mode = TransformMode.ENDURANCE_DECAY
        elif args.to_type == "enemy":
            args.transform_mode = TransformMode.ENDURANCE_ENEMY
        else:
            Assert.never()
    Assert.true(
        not (args.file and (args.minify == "off" or args.minify == "auto")),
        "--minify=off or --minify=auto is incompatible with --file"
    )
    Assert.true(
        not args.dry_run or args.file,
        "--dry-run must be used together with --file"
    )
    if args.file:
        args.minify_policy = MinifyPolicy.MAX
    else:
        if args.minify is None or args.minify == "auto":
            args.minify_policy = MinifyPolicy.ON_DEMAND
        elif args.minify == "off":
            args.minify_policy = MinifyPolicy.NO_MINIFY
        elif args.minify == "max":
            args.minify_policy = MinifyPolicy.MAX
        else:
            Assert.never()
    return args


def run_file(
        transform_mode: "TransformMode", minify_policy: "MinifyPolicy", dry_run: bool,
    ) -> None:
    with open(_INPUT_FILENAME, encoding="utf-8") as f:
        formula = f.read()
    engine = Engine(transform_mode, minify_policy, IoMode.FILE)
    formula = engine.run(formula)
    if dry_run:
        return
    with open(_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write(formula)


def run_repl(transform_mode: "TransformMode", minify_policy: "MinifyPolicy") -> None:
    print("Formula >")
    print("=" * 60)
    engine = Engine(transform_mode, minify_policy, IoMode.REPL)
    while True:
        formula = ""
        for line in sys.stdin:
            if line == ":quit\n":
                return
            elif line == ":clear\n":
                os.system("cls" if os.name == "nt" else "clear")
                print("Formula >")
                print("=" * 60)
                formula = ""
                continue
            formula += line
        formula = formula.strip()
        if len(formula) == 0:
            return
        try:
            formula = engine.run(formula)
        except ValueError:
            sys.stdout.write(traceback.format_exc())
            print("=" * 60)
            continue
        print("-" * 60)
        print(formula)
        print("=" * 60)


class Engine:
    def __init__(
            self,
            transform_mode: "TransformMode",
            minify_policy: "MinifyPolicy",
            io_mode: "IoMode",
        ) -> None:
        self._visitors = self._get_available_visitors(transform_mode)
        self._transformers = self._get_available_transformers(transform_mode, minify_policy)
        self._io_mode = io_mode

    def run(self, formula: str) -> str:
        start_time = time.perf_counter()
        results = []
        sub_formulas = self._split(formula)
        for sub_formula in sub_formulas:
            if len(sub_formula) > 0:
                sub_formula = self._unescape(sub_formula)
                for visitor in self._visitors:
                    visitor.run(sub_formula)
                for transformer in self._transformers:
                    sub_formula = transformer.run(sub_formula)
            results.append(sub_formula)
        end_time = time.perf_counter()
        self._print_summary(results, start_time, end_time)
        return "\n".join(results)

    def _get_available_visitors(
            self, transform_mode: "TransformMode"
        ) -> Iterable["Visitor"]:
        return [
            OperatorIdValidator(),
            OperatorElementalDamageValidator(),
            AnnotationValidator(transform_mode),
        ]

    def _get_available_transformers(
            self,
            transform_mode: "TransformMode",
            minify_policy: "MinifyPolicy"
        ) -> Iterable["Transformer"]:
        transformers = []
        transformers.extend(self._get_custom_pre_transformers(transform_mode))
        if transform_mode.from_damage():
            transformers.extend([
                PhysicalDamageBuffInjector(),
                MagicalDamageBuffInjector(),
                TrueDamageBuffInjector(),
                ElementalDamageBuffInjector(),
                DefaultElementalDamageBuffInjector(),
                InjuryBuffInjector(),
                AutomaticSkillBuffInjector(),
                OffensiveSkillAttackCountInjector(),
                OffensiveSkillInjector(),
                AttackSpanBuffInjector(),
                ManualAttackSpeedBuffInjector(),
            ])
        elif transform_mode.from_endurance():
            transformers.extend([
                PhysicalEnduranceBuffInjector(),
                MagicalEnduranceBuffInjector(),
                TrueEnduranceBuffInjector(),
                EnduranceFormulaGenerator(transform_mode),
            ])
        else:
            Assert.never()
        transformers.extend(self._get_custom_post_tranformers(transform_mode))
        transformers.extend([
            ReturnPredicateRemover(),
            BuffInvalidateWrapper(transform_mode),
            Minifier(minify_policy),
        ])
        return transformers

    def _get_custom_pre_transformers(
            self, transform_mode: "TransformMode"
        ) -> Sequence["Transformer"]:
        return []

    def _get_custom_post_tranformers(
            self, transform_mode: "TransformMode"
        ) -> Sequence["Transformer"]:
        if transform_mode.from_damage() and transform_mode.to_buff():
            formula_part_pairs = {}
        elif transform_mode.from_damage() and transform_mode.to_decay():
            formula_part_pairs = {
                "EnemyDefenseMajor": "B$1",
                "EnemyDefenseMinor": "B$1",
                "EnemyResistanceMajor": "B$2",
                "EnemyResistanceMinor": "B$2",
            }
        elif transform_mode.from_damage() and transform_mode.to_enemy():
            formula_part_pairs = {
                "EnemyDefenseMajor": "D$1",
                "EnemyDefenseMinor": "D$1",
                "EnemyResistanceMajor": "D$2",
                "EnemyResistanceMinor": "D$2",
                "EnemyWeightMajor": "D$3",
                "EnemyWeightMinor": "D$3",
                "EnemyRankMajor": "D$4",
                "EnemyRankMinor": "D$4",
                "EnemyMaxElementMajor": 'IF(D$4="领袖",2000,1000)',
                "EnemyMaxElementMinor": 'IF(D$4="领袖",2000,1000)',
                "EnemyAerial": "D$5",
            }
        elif transform_mode.from_endurance() and transform_mode.to_buff():
            formula_part_pairs = {}
        elif transform_mode.from_endurance() and transform_mode.to_decay():
            formula_part_pairs = {
                "EnemyDamagePerHit": "B$1",
            }
        elif transform_mode.from_endurance() and transform_mode.to_enemy():
            formula_part_pairs = {
                "EnemyDamagePerHit": "E$1",
                "EnemyAttackSpan": "E$2",
                "EnemyDamageType": "E$3",
                "EnemyRanged": "E$4",
            }
        else:
            Assert.never()
        result = []
        for key, value in formula_part_pairs.items():
            result.append(SingleWordReplacer(keyword=key, replacement=value))
        return result

    def _unescape(self, formula: str) -> str:
        if formula.startswith('"'):
            formula = re.sub(r'(?<!")"(?!")', '', formula)
            formula = re.sub(r'""', '"', formula)
        return formula

    def _split(self, formula: str) -> Iterable[str]:
        if self._io_mode == IoMode.FILE:
            return self._split_file(formula)
        elif self._io_mode == IoMode.REPL:
            return self._split_repl(formula)
        else:
            Assert.never()

    def _split_repl(self, formula: str) -> Iterable[str]:
        level = 0
        breakpoints = []
        for index, char in enumerate(formula):
            if char == "(":
                level += 1
            elif char == ")":
                level -= 1
                Assert.true(level >= 0, "Illegal formula with unpaired brackets")
            elif char == "=" and level == 0:
                breakpoints.append(index)
        Assert.true(level == 0, "Illegal formula with unpaired brackets")
        breakpoints.append(len(formula))
        if len(breakpoints) <= 2:
            yield formula.strip()
            return
        for index in range(len(breakpoints) - 1):
            this = breakpoints[index]
            next = breakpoints[index + 1]
            yield formula[this:next].strip()

    def _split_file(self, formula: str) -> Iterable[str]:
        pattern = re.compile(r'(?<!")"(?!")=.+?(?<!")"(?!")|^\n', re.MULTILINE | re.DOTALL)
        for match in re.finditer(pattern, formula):
            yield match.group().strip()

    def _print_summary(
            self, results: Sequence[str], start_time: float, end_time: float
        ) -> None:
        count_all = len(results)
        count_nonempty = sum(1 for item in results if len(item) > 0)
        if count_all <= 1:
            return
        if count_all == count_nonempty:
            message = (
                f"Processed {count_all} formulas "
                f"in {end_time - start_time:.2f} seconds")
        else:
            message = (
                f"Processed {count_nonempty} formulas and "
                f"{count_all - count_nonempty} empty lines in "
                f"{end_time - start_time:.2f} seconds"
            )
        print(message)


class Check:
    @staticmethod
    def not_none(groups: dict[str, str | None], field: str | Sequence[str]) -> bool:
        fields = [field] if isinstance(field, str) else field
        for key in fields:
            if groups[key] is None:
                return False
        return True

    @staticmethod
    def not_empty(groups: dict[str, str | None], field: str | Sequence[str]) -> bool:
        fields = [field] if isinstance(field, str) else field
        for key in fields:
            item = groups[key]
            if item is None or len(item) == 0:
                return False
        return True

    @staticmethod
    def equal(groups: dict[str, str | None], field1: str, field2: str) -> bool:
        return groups[field1] == groups[field2]
    
    @staticmethod
    def either(groups: dict[str, str | None], fields: Sequence[str]) -> bool:
        not_empty_count = 0
        for field in fields:
            if groups[field] is not None:
                not_empty_count += 1
        return not_empty_count == 1

    @staticmethod
    def only(groups: dict[str, str | None], field_groups: Sequence[Sequence[str]]) -> bool:
        not_none_counts = [0] * len(field_groups)
        for index, field_group in enumerate(field_groups):
            for field in field_group:
                if groups[field] is not None:
                    not_none_counts[index] += 1
        none_groups = 0
        success_groups = 0
        for index, count in enumerate(not_none_counts):
            if count == 0:
                none_groups += 1
            if count == len(field_groups[index]):
                success_groups += 1
        return success_groups == 1 and none_groups == len(field_groups) - 1
    
    @staticmethod
    def has_key(groups: dict[str, str | None], field: str | Sequence[str]) -> bool:
        fields = [field] if isinstance(field, str) else field
        for key in fields:
            if key not in groups:
                return False
        return True

    @staticmethod
    def has_value(
        groups: dict[str, str | None], field: str, value: str | Sequence[str]
    ) -> bool:
        values = [value] if isinstance(value, str) else value
        return groups[field] in values

    @staticmethod
    def is_number(groups: dict[str, str | None], field: str) -> bool:
        try:
            float(groups[field])  # type: ignore
        except ValueError:
            return False
        return True


class Assert:
    @classmethod
    def not_none(cls, groups: dict[str, str | None], field: str | Sequence[str]) -> None:
        if not Check.not_none(groups, field):
            raise ValueError(f"<{field}> must not be None\n{cls._dump(groups)}")

    @classmethod
    def not_empty(cls, groups: dict[str, str | None], field: str | Sequence[str]) -> None:
        if not Check.not_empty(groups, field):
            raise ValueError(f"<{field}> must not be empty\n{cls._dump(groups)}")

    @classmethod
    def equal(cls, groups: dict[str, str | None], field1: str, field2: str) -> None:
        if not Check.equal(groups, field1, field2):
            raise ValueError(f"<{field1}> must be equal to <{field2}>\n"
                             f"{cls._dump(groups)}")

    @classmethod
    def either(cls, groups: dict[str, str | None], fields: Sequence[str]) -> None:
        if not Check.either(groups, fields):
            raise ValueError(f"Expect exactly 1 field in <{fields}> to be not None\n"
                             f"{cls._dump(groups)}")

    @classmethod
    def only(
        cls, groups: dict[str, str | None], field_groups: Sequence[Sequence[str]]
    ) -> None:
        if not Check.only(groups, field_groups):
            raise ValueError(f"Expect exactly 1 field group in <{field_groups}>"
                             f" to be not None\n{cls._dump(groups)}")

    @classmethod
    def has_key(cls, groups: dict[str, str | None], field: str | Sequence[str]) -> None:
        if not Check.has_key(groups, field):
            raise ValueError(f"Expect matched group to contain key <{field}>\n"
                             f"{cls._dump(groups)}")

    @classmethod
    def has_value(
        cls, groups: dict[str, str | None], field: str, value: str | Sequence[str]
    ) -> None:
        if not Check.has_value(groups, field, value):
            raise ValueError(f"Expect <{field}> to have value <{value}>, "
                             f"got <{groups[field] if field in groups else None}>\n"
                             f"{cls._dump(groups)}")

    @classmethod
    def is_number(cls, groups: dict[str, str | None], field: str) -> None:
        if not Check.is_number(groups, field):
            raise ValueError(f"Expect <{field}> to be a number, "
                             f"got <{groups[field] if field in groups else None}>\n"
                             f"{cls._dump(groups)}")

    @classmethod
    def never(cls) -> None:
        raise RuntimeError("Control reaches unreachable code")

    @classmethod
    def fail(cls, error_message: str) -> None:
        raise ValueError(error_message)

    @classmethod
    def true(
        cls,
        predicate: bool,
        error_message: str,
        groups: dict[str, str | None] | None = None,
    ) -> None:
        if not predicate:
            message = (
                f"{error_message}\n{cls._dump(groups)}"
                if groups is not None
                else error_message
            )
            raise ValueError(message)

    @staticmethod
    def override(super_class: type) -> Callable[[Callable], Callable]:
        def override_checker(method: Callable) -> Callable:
            if method.__name__ not in dir(super_class):
                raise TypeError(f"Invalid override: `{super_class.__name__}."
                                f"{method.__name__}` does not exist")
            return method
        return override_checker

    @classmethod
    def _dump(cls, groups: dict[str, str | None]) -> str:
        return json.dumps(groups, indent=4)


class FormulaContext:
    def __init__(
            self,
            formula: str,
            operator_ids: Sequence[str],
            summon_ids: Sequence[str],
            elemental_type: "OperatorElementalType",
            length: Optional[int] = None,
            offsets: Optional[list[int]] = None,
            sealed_index:  Optional[int] = None,
        ) -> None:
        Assert.true((
            (length is None and offsets is None and sealed_index is None) or
            (
                length is not None and offsets is not None and sealed_index is not None and
                len(formula) >= length and
                len(offsets) == length + 1 and
                0 <= sealed_index <= length
            )
        ), "Illegal arguments")
        self._formula = formula
        self._operator_ids = tuple(operator_ids)
        self._summon_ids = tuple(summon_ids)
        self._operators = tuple(Operator(operator_id) for operator_id in operator_ids)
        self._summons = tuple(Summon(summon_id) for summon_id in summon_ids)
        self._subject_ids = tuple(set(
            any_id
            for any_ids in (operator_ids, summon_ids)
            for any_id in any_ids
        ))
        self._subjects = tuple(
            any_subject
            for any_subjects in (self._operators, self._summons)
            for any_subject in any_subjects
        )
        self._elemental_type = elemental_type
        self._length = len(formula) if length is None else length
        self._offsets = [0] * (self._length + 1) if offsets is None else offsets
        self._sealed_index = 0 if sealed_index is None else sealed_index

    @property
    def formula(self) -> str:
        return self._formula

    @property
    def operators(self) -> Sequence["Operator"]:
        return self._operators

    @property
    def summons(self) -> Sequence["Summon"]:
        return self._summons

    @property
    def subjects(self) -> Sequence[Union["Operator", "Summon"]]:
        return self._subjects

    @property
    def subject_ids(self) -> Sequence[str]:
        return self._subject_ids

    @property
    def elemental_type(self) -> "OperatorElementalType":
        return self._elemental_type

    def insert(self, text: str, index: int) -> None:
        Assert.true(0 <= index <= self._length, "Illegal index: overflow")
        aligned_index = self._shift(index, len(text), allow_shorten=False)
        self._formula = self._formula[:aligned_index] + text + self._formula[aligned_index:]
        if _DEBUG:
            print(f"Insert `{text}` at {index}")
            print(self._formula)
            print("-" * 30)

    def replace(self, text: str, start: int, end: int, allow_shorten: bool) -> None:
        Assert.true(0 <= start <= end <= self._length, "Illegal index: overflow")
        old_length = end - start
        new_length = len(text)
        aligned_start, aligned_end = self.peek((start, end))
        self._shift(start, new_length - old_length, allow_shorten)
        self._formula = self._formula[:aligned_start] + text + self._formula[aligned_end:]
        if _DEBUG:
            print(f"Replace with `{text}` at {start}:{end}")
            print(self._formula)
            print("-" * 30)

    def peek(self, index: tuple[int, int]) -> tuple[int, int]:
        results = []
        for item in index:
            results.append(self._peek(item))
        return tuple(results)

    def get(self, match: re.Match) -> str:
        aligned_start, aligned_end = self.peek(match.span())
        return self.formula[aligned_start:aligned_end]

    def detach(self) -> "FormulaContext":
        return FormulaContext(
            formula=self._formula,
            operator_ids=self._operator_ids,
            summon_ids=self._summon_ids,
            elemental_type=self._elemental_type,
            length=self._length,
            offsets=list(self._offsets),
            sealed_index=self._sealed_index,
        )

    def _shift(self, index: int, length: int, allow_shorten: bool) -> int:
        Assert.true(0 <= index <= self._length, "Illegal index: overflow")
        Assert.true(allow_shorten or length > 0, f"Expect shift length > 0, got {length}")
        aligned_index = index + self._offsets[index]
        for sub in range(index, self._length + 1):
            self._offsets[sub] += length
        self._seal(index)
        return aligned_index

    def _peek(self, index: int) -> int:
        Assert.true(0 <= index <= self._length, "Illegal index: overflow")
        return index + self._offsets[index]

    def _seal(self, index: int) -> None:
        Assert.true(index >= self._sealed_index, 
                    "Attempting to re-seal an already sealed index in formula")
        self._sealed_index = index


class MatchContext:
    def __init__(
            self,
            match: re.Match,
            subject: Union["Operator", "Summon", None],
            parent: FormulaContext,
        ) -> None:
        self._match = match
        self._subject = subject
        self._parent = parent

    @property
    def formula(self) -> str:
        return self._parent.formula

    @property
    def match(self) -> re.Match:
        return self._match

    @property
    def subject(self) -> Union["Operator", "Summon", None]:
        return self._subject

    @property
    def operators(self) -> Sequence["Operator"]:
        return self._parent.operators

    @property
    def summons(self) -> Sequence["Summon"]:
        return self._parent.summons

    @property
    def subject_ids(self) -> Sequence[str]:
        return self._parent.subject_ids

    @property
    def elemental_type(self) -> "OperatorElementalType":
        return self._parent.elemental_type

    def insert(
        self,
        text: str,
        *,
        before: Optional[str] = None,
        after: Optional[str] = None,
        at: Optional[int] = None,
    ) -> None:
        Assert.true(
            (
                (before is not None and after is None and at is None) or 
                (before is None and after is not None and at is None) or
                (before is None and after is None and at is not None)
            ),
            "Expect exactly one argument of `before`, `after`, and `at` is given"
        )
        if before is not None:
            self._parent.insert(text, self._match.start(before))
        elif after is not None:
            self._parent.insert(text, self._match.end(after))
        elif at is not None:
            self._parent.insert(text, at)
        else:
            Assert.never()

    def replace(self, text: str, subpattern_name: str, allow_shorten: bool = False) -> None:
        start, end = self._match.span(subpattern_name)
        self._parent.replace(text, start, end, allow_shorten)

    def get(self, subpattern_name: str) -> str:
        aligned_start, aligned_end = self._parent.peek(self._match.span(subpattern_name))
        return self._parent.formula[aligned_start:aligned_end]

    def get_all(self) -> str:
        aligned_start, aligned_end = self._parent.peek(self._match.span())
        return self._parent.formula[aligned_start:aligned_end]

    def detach(self) -> "MatchContext":
        return MatchContext(self._match, self._subject, self._parent.detach())

    def ensure_subject_exist(self) -> None:
        if self.subject is None:
            Assert.fail(f"Expect a subject in {self.get_all()}")


class TransformMode(Enum):
    DAMAGE_BUFF = "DamageBuff"
    DAMAGE_DECAY = "DamageDecay"
    DAMAGE_ENEMY = "DamageEnemy"
    ENDURANCE_BUFF = "EnduranceBuff"
    ENDURANCE_DECAY = "EnduranceDecay"
    ENDURANCE_ENEMY = "EnduranceEnemy"
    
    def from_damage(self) -> bool:
        return self in (
            TransformMode.DAMAGE_BUFF,
            TransformMode.DAMAGE_DECAY,
            TransformMode.DAMAGE_ENEMY,
        )

    def from_endurance(self) -> bool:
        return self in (
            TransformMode.ENDURANCE_BUFF,
            TransformMode.ENDURANCE_DECAY,
            TransformMode.ENDURANCE_ENEMY,
        )

    def to_buff(self) -> bool:
        return self in (
            TransformMode.DAMAGE_BUFF,
            TransformMode.ENDURANCE_BUFF,
        )

    def to_decay(self) -> bool:
        return self in (
            TransformMode.DAMAGE_DECAY,
            TransformMode.ENDURANCE_DECAY,
        )

    def to_enemy(self) -> bool:
        return self in (
            TransformMode.DAMAGE_ENEMY,
            TransformMode.ENDURANCE_ENEMY,
        )


class MinifyPolicy(Enum):
    NO_MINIFY = "NoMinify"
    ON_DEMAND = "OnDemand"
    MAX = "Max"


class BuffJoinPolicy(Enum):
    MAX = "Max"
    MIN = "Min"
    ADD = "Add"
    MULTIPLY = "Multiply"

    @property
    def function(self) -> str:
        if self == BuffJoinPolicy.MAX:
            return "MAX"
        elif self == BuffJoinPolicy.MIN:
            return "MIN"
        elif self in (BuffJoinPolicy.ADD, BuffJoinPolicy.MULTIPLY):
            return ""
        else:
            Assert.never()

    @property
    def separator(self) -> str:
        if self in (BuffJoinPolicy.MAX, BuffJoinPolicy.MIN):
            return ","
        elif self == BuffJoinPolicy.ADD:
            return "+"
        elif self == BuffJoinPolicy.MULTIPLY:
            return "*"
        else:
            Assert.never()


class IoMode(Enum):
    FILE = "File"
    REPL = "Repl"


class OperatorProfession(Enum):
    VANGUARD = "Vanguard"
    GUARD = "Guard"
    DEFENDER = "Defender"
    SNIPER = "Sniper"
    CASTER = "Caster"
    MEDIC = "Medic"
    SUPPORTER = "Supporter"
    SPECIALIST = "Specialist"


class OperatorPosition(Enum):
    MELEE = "Melee"
    RANGED = "Ranged"
    ANY = "Any"


class SummonPosition(Enum):
    MELEE = "Melee"
    RANGED = "Ranged"
    ANY = "Any"
    MULTIPLE = "Multiple"
    NONE = "None"


class PlainPosition(Enum):
    MELEE = "Melee"
    RANGED = "Ranged"


class OperatorElementalType(Enum):
    NONE = "None"
    FIRE = "Fire"
    DARK = "Dark"


class OperatorInfo:
    def __init__(
        self,
        identifier: str,
        name: str,
        profession: str,
        position: str,
        elemental_type: str,
        block_count: str,
        summon_attack: str,
        summon_position: str,
        summon_block_count: str,
    ) -> None:
        self.identifier = identifier
        self.name = name
        self.profession = self._parse_profession(profession)
        self.position = self._parse_position(position)
        self.elemental_type = self._parse_elemental_type(elemental_type)
        self.block_count: int = self._parse_block_count(block_count, required=True)
        self.has_summon = self._parse_has_summon(summon_attack)
        self.summon_position = self._parse_summon_position(summon_position)
        self.summon_block_count = self._parse_block_count(summon_block_count, required=False)

    def _parse_profession(self, profession: str) -> OperatorProfession:
        profession_map = {
            "先锋": OperatorProfession.VANGUARD,
            "近卫": OperatorProfession.GUARD,
            "重装": OperatorProfession.DEFENDER,
            "狙击": OperatorProfession.SNIPER,
            "术师": OperatorProfession.CASTER,
            "医疗": OperatorProfession.MEDIC,
            "辅助": OperatorProfession.SUPPORTER,
            "特种": OperatorProfession.SPECIALIST,
        }
        Assert.true(
            profession in profession_map, f"Invalid operator profession: {profession}"
        )
        return profession_map[profession]

    def _parse_position(self, position: str) -> OperatorPosition:
        position_map = {
            "地面": OperatorPosition.MELEE,
            "高台": OperatorPosition.RANGED,
            "任意": OperatorPosition.ANY,
        }
        Assert.true(
            position in position_map, f"Invalid operator position: {position}"
        )
        return position_map[position]

    def _parse_elemental_type(self, elemental_type: str) -> OperatorElementalType:
        elemental_type_map = {
            "": OperatorElementalType.NONE,
            "灼燃": OperatorElementalType.FIRE,
            "凋亡": OperatorElementalType.DARK,
        }
        Assert.true(
            elemental_type in elemental_type_map,
            f"Invalid operator elemental type: {elemental_type}"
        )
        return elemental_type_map[elemental_type]

    def _parse_has_summon(self, summon_attack: str) -> bool:
        # summon_attack must be "", "#N/A", or a positive integer
        if summon_attack == "":
            return False
        elif summon_attack == "#N/A":
            return True
        else:
            try:
                int(summon_attack)
            except ValueError:
                raise
            return True

    def _parse_summon_position(self, position: str) -> SummonPosition:
        position_map = {
            "": SummonPosition.NONE,
            "地面": SummonPosition.MELEE,
            "高台": SummonPosition.RANGED,
            "任意": SummonPosition.ANY,
            "多种": SummonPosition.MULTIPLE,
        }
        Assert.true(
            position in position_map, f"Invalid summon position: {position}"
        )
        return position_map[position]

    def _parse_block_count(self, block_count: str, required: bool) -> int | None:
        # block_count must be "", "#N/A", or a non-negative integer
        if block_count in ("", "#N/A"):
            Assert.true(not required, "Illegal block count with empty or NA value")
            return None
        try:
            block_count_number = int(block_count)
        except ValueError:
            raise
        Assert.true(0 <= block_count_number <= 4, f"Illegal block count {block_count}")
        return block_count_number


class DeployableUnit:
    def is_operator(self) -> bool:
        return isinstance(self, Operator)

    def is_summon(self) -> bool:
        return isinstance(self, Summon)


class Operator(DeployableUnit):
    _info: dict[str, OperatorInfo] | None = None

    def __init__(self, identifier: str) -> None:
        if Operator._info is None:
            Operator._load_operator_info()
        Assert.true(
            identifier in Operator._info, f"Operator ID {identifier} not found"
        )
        self._id = identifier

    @classmethod
    def _load_operator_info(cls) -> None:
        result = {}
        with open(_OPERATOR_INFO_FILENAME, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row["ID"]
                name = row["干员名称"]
                profession = row["职业"]
                position = row["位置"]
                elemental_type = row["元素类型"]
                block_count = row["阻挡数"]
                summon_attack = row["召唤物攻击"]
                summon_position = row["召唤物位置"]
                summon_block_count = row["召唤物阻挡数"]
                result[key] = OperatorInfo(
                    key, name, profession, position, elemental_type, block_count,
                    summon_attack, summon_position, summon_block_count,
                )
        cls._info = result

    @property
    def identifier(self) -> str:
        return Operator._info[self._id].identifier

    @property
    def name(self) -> str:
        return Operator._info[self._id].name

    def is_vanguard(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.VANGUARD

    def is_guard(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.GUARD

    def is_defender(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.DEFENDER

    def is_sniper(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.SNIPER

    def is_caster(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.CASTER

    def is_medic(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.MEDIC

    def is_supporter(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.SUPPORTER

    def is_specialist(self) -> bool:
        return Operator._info[self._id].profession == OperatorProfession.SPECIALIST

    def is_melee(self) -> bool:
        return Operator._info[self._id].position in (
            OperatorPosition.MELEE, OperatorPosition.ANY
        )

    def is_ranged(self) -> bool:
        return Operator._info[self._id].position in (
            OperatorPosition.RANGED, OperatorPosition.ANY
        )

    def is_ranged_strict(self) -> bool:
        return Operator._info[self._id].position == OperatorPosition.RANGED

    def is_selectable(self):
        return True

    def block_count(self) -> int:
        return Operator._info[self._id].block_count

    def has_summon(self) -> bool:
        return Operator._info[self._id].has_summon

    def deal_elemental(self) -> bool:
        return Operator._info[self._id].elemental_type != OperatorElementalType.NONE

    def deal_elemental_fire(self) -> bool:
        return Operator._info[self._id].elemental_type == OperatorElementalType.FIRE

    def deal_elemental_dark(self) -> bool:
        return Operator._info[self._id].elemental_type == OperatorElementalType.DARK

    def get_profession(self) -> str:
        return Operator._info[self._id].profession.value

    def is_summon_melee(self) -> bool:
        return Operator._info[self._id].summon_position in (
            SummonPosition.MELEE, SummonPosition.ANY
        )

    def is_summon_ranged(self) -> bool:
        return Operator._info[self._id].summon_position in (
            SummonPosition.RANGED, SummonPosition.ANY
        )

    def is_summon_ranged_strict(self) -> bool:
        return Operator._info[self._id].summon_position == SummonPosition.RANGED

    def is_summon_multiple_positions(self) -> bool:
        return Operator._info[self._id].summon_position == SummonPosition.MULTIPLE

    def summon_block_count(self) -> int:
        info = Operator._info[self._id]
        Assert.true(
            info.summon_block_count is not None,
            f"Expect {info.name}'s summon block count to be not None"
        )
        return info.summon_block_count


class Summon(DeployableUnit):
    def __init__(self, identifier: str) -> None:
        if Operator._info is None:
            Operator._load_operator_info()
        Assert.true(
            identifier in Operator._info, f"Operator ID {identifier} not found"
        )
        self._id = identifier
        self._parent = Operator(self._id)
        Assert.true(
            self._parent.has_summon(),
            f"Illegal summons: {self._parent.name} does not have summons"
        )

    @property
    def parent(self) -> Operator:
        return self._parent

    @property
    def identifier(self) -> str:
        return self.parent.identifier

    def is_melee(self) -> bool:
        return self.parent.is_summon_melee()

    def is_ranged(self) -> bool:
        return self.parent.is_summon_ranged()

    def is_ranged_strict(self) -> bool:
        return self.parent.is_summon_ranged_strict()

    def has_multiple_positions(self) -> bool:
        return self.parent.is_summon_multiple_positions()

    def is_selectable(self) -> bool:
        raise NotImplementedError

    def block_count(self) -> int:
        return self.parent.summon_block_count()


class AnnotatedBuff:
    def __init__(
        self,
        key: str,
        value: str,
        join_policy: BuffJoinPolicy = BuffJoinPolicy.MAX,
    ):
        Assert.true(re.fullmatch("@[A-Za-z]+", key) is not None, "Illegal annotation key")
        self.key = key
        self.value = value
        self.join_policy = join_policy


class Visitor:
    def run(self, formula: str) -> None:
        raise NotImplementedError


class OperatorIdValidator(Visitor):
    @Assert.override(Visitor)
    def run(self, formula: str) -> None:
        pattern = re.compile(_operator_or_summon_id_regex, re.M | re.X)
        matches = re.finditer(pattern, formula)
        for match in matches:
            return
        Assert.fail("Expect operator id in formula")


class OperatorElementalDamageValidator(Visitor):
    @Assert.override(Visitor)
    def run(self, formula: str) -> None:
        id_pattern = re.compile(_operator_or_summon_id_regex, re.M | re.X)
        matches = re.finditer(id_pattern, formula)
        allow_elemental = False
        operator_names = []
        for match in matches:
            operator_id = match.group("id")
            operator = Operator(operator_id)
            operator_names.append(operator.name)
            if operator.deal_elemental():
                allow_elemental = True
                break
        if not allow_elemental:
            patterns = [
                re.compile(_elemental_damage_regex, re.M | re.X),
                re.compile(_default_elemental_damage_regex, re.M | re.X),
                re.compile(_injury_regex, re.M | re.X),
            ]
            for pattern in patterns:
                matches = re.findall(pattern, formula)
                Assert.true(
                    len(matches) == 0,
                    f"Illegal elemental damage: {", ".join(operator_names)} "
                    f"is not expected to deal elemental damage"
                )


class AnnotationValidator(Visitor):
    def __init__(self, transform_mode: TransformMode) -> None:
        self._transform_mode = transform_mode
        self._annotation_key_regex = re.compile(_annotation_key_regex, re.M | re.X)

    @Assert.override(Visitor)
    def run(self, formula: str) -> None:
        if self._transform_mode.from_damage():
            self._ensure_legal(formula, legal_annotations=[
                "@MonoEncouragedAttack", "@MonoEnergizedAttack",
                "@MonoEnemyStolenDefense", "@MonoEnemyFrozenResistance",
                "@MonoEnemyVulnerable", "@MonoEnemyVulnerablePhysical",
                "@MonoEnemyVulnerableMagical", "@MonoEnemyVulnerableElemental",
                "@MonoSkillPointAutomatic",
                "@True", "@InjuryFire", "@InjuryDark", "@SkillOffensiveAttackCount",
                "@AttackSpeed", "@SkillAutomatic", "@SkillOffensive",
            ])
        elif self._transform_mode.from_endurance():
            self._ensure_legal(formula, legal_annotations=[
                "@MonoEncouragedHealth", "@MonoEncouragedDefense", "@MonoShelter",
                "@MonoEnemyStolenAttack", "@MonoEnemyWeakenedAttack",
                "@MonoEnemyStolenAttackSpeed", "@MonoEnemyColdAttackSpeed",
                "@CrystalBarrier", "@ProjectileRemoval", "@DamageLoss", "@Evasion",
                "@EnemyAttackSpeedLoss", "@BlockCount",
                "@PositionMelee", "@PositionRanged",
            ])
        else:
            Assert.never()

    def _ensure_legal(self, formula: str, legal_annotations: Sequence[str]) -> None:
        for match in re.finditer(self._annotation_key_regex, formula):
            groups = match.groupdict()
            Assert.has_key(groups, "key")
            Assert.has_value(groups, "key", legal_annotations)


class Transformer:
    def run(self, formula: str) -> str:
        formula_context = self._new_formula_context(formula)
        matches = re.finditer(self._get_pattern(), formula)
        for match in matches:
            self._ensure_valid(match)
            match_context = self._new_match_context(match, parent=formula_context)
            for transform_pass in self._get_passes():
                transform_pass(match_context)
        self._ensure_legal_brackets(formula_context.formula)
        return formula_context.formula

    def _run_and_extract_once(self, formula: str, subpattern_name: str) -> str:
        formula_context = self._new_formula_context(formula)
        matches = list(re.finditer(self._get_pattern(), formula))
        Assert.true(len(matches) == 1, f"Expect 1 match, got {len(matches)}")
        self._ensure_valid(matches[0])
        match_context = self._new_match_context(matches[0], parent=formula_context)
        for transform_pass in self._get_passes():
            transform_pass(match_context)
        self._ensure_legal_brackets(formula_context.formula)
        return match_context.get(subpattern_name)

    def _get_pattern(self) -> re.Pattern:
        raise NotImplementedError

    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        raise NotImplementedError

    def _ensure_valid(self, match: re.Match) -> None:
        raise NotImplementedError

    def __init__(self):
        self._operator_id_pattern = re.compile(_operator_id_regex, re.M | re.X)
        self._summon_id_pattern = re.compile(_summon_id_regex, re.M | re.X)

    def _new_formula_context(self, formula: str) -> FormulaContext:
        return FormulaContext(
            formula=formula,
            operator_ids=self._get_operator_ids(formula),
            summon_ids=self._get_summon_ids(formula),
            elemental_type=self._get_elemental_type(formula),
        )

    def _new_match_context(self, match: re.Match, parent: FormulaContext) -> MatchContext:
        return MatchContext(
            match=match,
            subject=self._get_subject(parent, match),
            parent=parent,
        )

    def _get_operator_ids(self, formula: str) -> Sequence[str]:
        return tuple(set(self._get_ids(formula, self._operator_id_pattern)))

    def _get_summon_ids(self, formula: str) -> Sequence[str]:
        return tuple(set(self._get_ids(formula, self._summon_id_pattern)))

    def _get_ids(self, formula: str, pattern: re.Pattern) -> Iterable[str]:
        matches = re.finditer(pattern, formula)
        for match in matches:
            yield match.group("id")

    def _get_operators(self, formula: str) -> Sequence[Operator]:
        return tuple(Operator(id) for id in self._get_operator_ids(formula))

    def _get_subject(
            self, context: FormulaContext, match: re.Match
        ) -> Operator | Summon | None:
        """
        Get the subject within the current match span - can be an operator, a summon or None.
        None means we cannot determine the subject of the current match because the
        information is locally missing and the global formula has more than one subject.
        """
        formula_slice = context.get(match)
        operator_ids = self._get_operator_ids(formula_slice)
        summon_ids = self._get_summon_ids(formula_slice)
        if len(operator_ids) == 1 and len(summon_ids) == 0:
            return Operator(operator_ids[0])
        elif len(operator_ids) == 0 and len(summon_ids) == 1:
            return Summon(summon_ids[0])
        else:
            # either subjects missing in the current match (for small-span transformers)
            # or too many subjects to determine which to use (for large-span transformers)
            # try to use subjects from the global formula if only one exists
            if len(context.subjects) >= 2:
                return None
            elif len(context.subjects) == 1:
                return context.subjects[0]
            else:
                Assert.never()  # expect at least one subject

    def _get_elemental_type(self, formula: str) -> OperatorElementalType:
        operators = self._get_operators(formula)
        error_message = (
            f"Expect {", ".join([op.name for op in operators])} to deal "
            "only one kind of elemental damage"
        )
        has_elemental = False
        fire = False
        dark = False
        for operator in operators:
            if operator.deal_elemental():
                has_elemental = True
            if operator.deal_elemental_fire():
                Assert.true(not dark, error_message)
                fire = True
            elif operator.deal_elemental_dark():
                Assert.true(not fire, error_message)
                dark = True
        if has_elemental:
            if fire:
                return OperatorElementalType.FIRE
            elif dark:
                return OperatorElementalType.DARK
            else:
                Assert.never()
        else:
            return OperatorElementalType.NONE

    def _ensure_legal_brackets(self, formula: str) -> None:
        level = 0
        for char in formula:
            if char == "(":
                level += 1
            elif char == ")":
                level -= 1
                Assert.true(
                    level >= 0, 
                    "Illegal unpaired brackets with too many `)` in formula"
                )
        Assert.true(
            level == 0,
            "Illegal unpaired brackets with too many `(` in formula"
        )


class BuffInjector(Transformer):
    def __init__(self):
        self._annotation_pattern = re.compile(_annotation_regex, re.M | re.X)
        self._annotations_pattern = re.compile(_annotations_regex, re.M | re.X)
        super().__init__()

    def _has_annotated_buffs(self, subpattern_name: str, context: MatchContext) -> bool:
        return re.search(
            self._annotation_pattern, context.get(subpattern_name)
        ) is not None

    def _ensure_all_buffs_annotated(
            self, subpatten_name: str, context: MatchContext,
        ) -> None:
        formula_slice = context.get(subpatten_name)
        Assert.true(
            re.fullmatch(self._annotations_pattern, formula_slice) is not None,
            f"Expect all buffs to be annotated in `{formula_slice}`"
        )

    def _inject_annotated_buffs(
        self,
        annotated_buffs: AnnotatedBuff | Iterable[AnnotatedBuff],
        *,
        separator: Literal["+", "-", "*", "/"] = "+",
        subpattern_name: str,
        context: MatchContext,
    ) -> None:
        annotated_buffs = (
            (annotated_buffs,)
            if isinstance(annotated_buffs, AnnotatedBuff)
            else annotated_buffs
        )
        annotated_buff_map = {buff.key: buff for buff in annotated_buffs}
        start = context.match.start(subpattern_name)
        matches = re.finditer(
            self._annotation_pattern, context.get(subpattern_name)
        )
        for match in matches:
            groups = match.groupdict()
            Assert.has_key(groups, ["key", "value"])
            Assert.has_value(groups, "key", annotated_buff_map.keys())
            Assert.is_number(groups, "value")
            key = groups["key"]
            buff = annotated_buff_map[key]
            Assert.true(buff.value is not None, "Illegal annotated buff with None value")
            value_start = start + match.start("value")
            value_end = start + match.end("value")
            context.insert(f'{buff.join_policy.function}(', at=value_start)
            context.insert(f'{buff.join_policy.separator}{buff.value})', at=value_end)
            del annotated_buff_map[key]
        for buff in annotated_buff_map.values():
            context.insert(
                f'{separator}(N("{buff.key}")+{buff.join_policy.function}({buff.value}))',
                after=subpattern_name
            )


class DamageBuffInjector(BuffInjector):
    def __init__(self):
        super().__init__()

    def _inject_attack_first_value(self, context: MatchContext) -> None:
        context.insert("+BuffDamageAttackFirstValue", after="attack_first_value")

    def _inject_attack_first_ratio(self, context: MatchContext) -> None:
        context.ensure_subject_exist()
        buff = '+BuffDamageAttackFirstRatio'
        if context.subject.is_melee():
            buff += '+BuffDamageAttackFirstRatioMelee'
        if context.subject.is_operator():
            profession = context.subject.get_profession()
            buff += f'+BuffDamageAttackFirstRatio{profession}'
        mono_buff = (
            'BuffDamageMonoEnergizedAttackFirstRatio'
            ',BuffDamageMonoEnergizedAttackFirstRatioMelee'
            if context.subject.is_melee()
            else 'BuffDamageMonoEnergizedAttackFirstRatio'
        )
        if context.match.group("attack_first_ratio") is not None:
            self._inject_annotated_buffs(
                AnnotatedBuff("@MonoEnergizedAttack", mono_buff),
                subpattern_name="attack_first_ratio",
                context=context,
            )
            context.insert(buff, after="attack_first_ratio")
        else:
            context.insert(
                f'*(1+(N("@MonoEnergizedAttack")+MAX({mono_buff})){buff})',
                after="attack_first_ratio_empty"
            )

    def _inject_attack_final_value(self, context: MatchContext) -> None:
        buff = '+BuffDamageAttackFinalValue'
        mono_buff = 'BuffDamageMonoEncouragedAttackFinalValue'
        self._inject_annotated_buffs(
            AnnotatedBuff("@MonoEncouragedAttack", mono_buff),
            subpattern_name="attack_final_value",
            context=context,
        )
        context.insert(buff, after="attack_final_value")

    def _get_attack_gain_pass(
        self, damage_type: Literal["physical", "magical", "true", "elemental"]
    ) -> Callable[[MatchContext], None]:
        buffs = {
            "physical": "+BuffDamagePhysicalGainValue",
            "magical": "+BuffDamageMagicalGainValue",
            "true": "+BuffDamageTrueGainValue",
            "elemental": "+BuffDamageElementalGainValue",
        }
        Assert.true(damage_type in buffs, f"Illegal damage_type `{damage_type}`")
        return lambda context: context.insert(buffs[damage_type], after="attack_gain")

    def _get_damage_weakener_pass(
        self, damage_type: Literal["physical", "magical"], allow_empty: bool = False
    ) -> Callable[[MatchContext], None]:
        if damage_type == "physical":
            buff = "+BuffDamageEnemyDefenseLossValue"
            mono_buffs = {
                "@MonoEnemyStolenDefense": "BuffDamageMonoEnemyStolenDefenseLossValue"
            }
            final_buff = "*BuffDamageEnemyDefenseLossFinalRatio"
            weakener_ungrouped = "defense_ungrouped"
            weakener_loss_value = "defense_loss_value"
            weakener_loss_value_ungrouped = "defense_loss_value_ungrouped"
            weakener_loss_value_empty = "defense_loss_value_empty"
            weakener_loss_ratio = "defense_loss_ratio"
        elif damage_type == "magical":
            buff = "+BuffDamageEnemyResistanceLossValue"
            mono_buffs = {
                "@MonoEnemyFrozenResistance": "BuffDamageMonoEnemyFrozenResistanceLossValue"
            }
            final_buff = "*BuffDamageEnemyResistanceLossFinalRatio"
            weakener_ungrouped = "resist_ungrouped"
            weakener_loss_value = "resist_loss_value"
            weakener_loss_value_ungrouped = "resist_loss_value_ungrouped"
            weakener_loss_value_empty = "resist_loss_value_empty"
            weakener_loss_ratio = "resist_loss_ratio"
        else:
            Assert.never()
        def inject_weakener(context: MatchContext) -> None:
            if context.match.group(weakener_ungrouped) is None:
                if context.match.group(weakener_loss_value) is not None:
                    self._inject_annotated_buffs(
                        (AnnotatedBuff(key, value) for key, value in mono_buffs.items()),
                        subpattern_name=weakener_loss_value,
                        context=context,
                    )
                    context.insert(buff, after=weakener_loss_value)
                elif context.match.group(weakener_loss_value_ungrouped) is not None:
                    context.insert('(', before=weakener_loss_value_ungrouped)
                    for key, value in mono_buffs.items(): 
                        context.insert(
                            f'+(N("{key}")+MAX({value}))',
                            after=weakener_loss_value_ungrouped
                        )
                    context.insert(f'{buff})', after=weakener_loss_value_ungrouped)
                elif context.match.group(weakener_loss_value_empty) is not None:
                    context.insert('-(0', after=weakener_loss_value_empty)
                    for key, value in mono_buffs.items():
                        context.insert(
                            f'+(N("{key}")+MAX({value}))',
                            after=weakener_loss_value_empty
                        )
                    context.insert(f'{buff})', after=weakener_loss_value_empty)
                else:
                    Assert.true(allow_empty, "Expect weakener to be not empty")
                if context.match.group(weakener_loss_ratio) is not None:
                    context.insert(final_buff, after=weakener_loss_ratio)
                else:
                    Assert.true(allow_empty, "Expect weakener to be not empty")
            else:
                context.insert('MAX(((', before=weakener_ungrouped)
                context.insert('-(0', after=weakener_ungrouped)
                for key, value in mono_buffs.items():
                    context.insert(
                        f'+(N("{key}")+MAX({value}))',
                        after=weakener_ungrouped
                    )
                context.insert(f'{buff})){final_buff}),0)', after=weakener_ungrouped)
        return inject_weakener

    def _get_attack_final_ratio_pass(
        self,
        damage_type: Literal["physical", "magical", "true", "elemental"],
        allow_empty: bool = False,
    ) -> Callable[[MatchContext], None]:
        if damage_type == "physical":
            buff = '*BuffDamagePhysicalFinalRatio'
            mono_buffs = {
                "@MonoEnemyVulnerable": "BuffDamageMonoEnemyVulnerableFinalRatio",
                "@MonoEnemyVulnerablePhysical": "BuffDamageMonoEnemyVulnerablePhysicalFinalRatio",
            }
        elif damage_type == "magical":
            buff = '*BuffDamageMagicalFinalRatio'
            mono_buffs = {
                "@MonoEnemyVulnerable": "BuffDamageMonoEnemyVulnerableFinalRatio",
                "@MonoEnemyVulnerableMagical": "BuffDamageMonoEnemyVulnerableMagicalFinalRatio",
            }
        elif damage_type == "true":
            buff = '*BuffDamageTrueFinalRatio'
            mono_buffs = {
                "@MonoEnemyVulnerable": "BuffDamageMonoEnemyVulnerableFinalRatio",
            }
        elif damage_type == "elemental":
            buff = '*BuffDamageElementalFinalRatio'
            mono_buffs = {
                "@MonoEnemyVulnerableElemental": None,  # incomplete buff
            }
        else:
            Assert.never()
        def _get_mono_enemy_vulnerable_buff(elemental_type: OperatorElementalType) -> str:
            Assert.true(
                elemental_type != OperatorElementalType.NONE,
                "Expect operator to deal elemental damage"
            )
            mono_buff = "BuffDamageMonoEnemyVulnerableElementalFinalRatio"
            if elemental_type == OperatorElementalType.FIRE:
                mono_buff += ",BuffDamageMonoEnemyVulnerableElementalFinalRatioFire"
            elif elemental_type == OperatorElementalType.DARK:
                mono_buff += ",BuffDamageMonoEnemyVulnerableElementalFinalRatioDark"
            else:
                Assert.never()
            return mono_buff
        def _complete_mono_buffs(
            mono_buffs: dict[str, Optional[str]], elemental_type: OperatorElementalType
        ) -> dict[str, str]:
            for key in mono_buffs:
                if mono_buffs[key] is None:
                    Assert.true(
                        key == "@MonoEnemyVulnerableElemental",
                        f"Unsupported empty mono buff key: {key}"
                    )
                    mono_buffs[key] = _get_mono_enemy_vulnerable_buff(elemental_type)
            return mono_buffs
        def _inject_attack_final_ratio(context: MatchContext) -> None:
            completed_mono_buffs = _complete_mono_buffs(mono_buffs, context.elemental_type)
            if context.match.group("damage_final_ratio") is not None:
                self._inject_annotated_buffs(
                    (
                        AnnotatedBuff(key, value)
                        for key, value in completed_mono_buffs.items()
                    ),
                    separator="*",
                    subpattern_name="damage_final_ratio",
                    context=context,
                )
                context.insert(buff, after="damage_final_ratio")
            elif context.match.group("damage_final_ratio_empty") is not None:
                context.insert('*(1', after="damage_final_ratio_empty")
                for key, value in completed_mono_buffs.items():
                    context.insert(
                        f'*(N("{key}")+MAX({value}))', after="damage_final_ratio_empty"
                    )
                context.insert(f'{buff})', after="damage_final_ratio_empty")
            else:
                Assert.true(allow_empty, "Expect damage final ratio to be not empty")
        return _inject_attack_final_ratio


class PhysicalDamageBuffInjector(DamageBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_physical_damage_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_first_value,
            self._inject_attack_first_ratio,
            self._inject_attack_final_value,
            self._get_attack_gain_pass(damage_type="physical"),
            self._inject_physical_line_2,
            self._get_damage_weakener_pass(damage_type="physical"),
            self._get_attack_final_ratio_pass(damage_type="physical"),
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "physical_line_1", "id", "attack_first_value", "attack_first_ratio",
            "attack_first_ratio_empty", "attack_final_value", "attack_multiplier",
            "attack_gain", "physical_line_2", "defense_loss_value",
            "defense_loss_value_ungrouped", "defense_loss_value_empty",
            "defense_loss_ratio", "defense_ignore_value",
            "defense_ignore_value_ungrouped", "defense_ignore_value_empty",
            "defense_ignore_ratio", "defense_ungrouped", "damage_final_ratio",
            "damage_final_ratio_empty",
        ])
        Assert.not_none(groups, [
            "attack_first_value", "attack_final_value", 
            "attack_multiplier", "attack_gain",
        ])
        Assert.not_empty(groups, ["physical_line_1", "physical_line_2", "id"])
        Assert.equal(groups, "physical_line_1", "physical_line_2")
        Assert.either(groups, ["attack_first_ratio", "attack_first_ratio_empty"])
        Assert.true(
            (
                Check.not_none(groups, "defense_ungrouped")
            ) or (
                Check.either(groups, [
                    "defense_loss_value",
                    "defense_loss_value_ungrouped",
                    "defense_loss_value_empty",
                ]) and
                Check.not_none(groups, "defense_loss_ratio") and
                Check.either(groups, [
                    "defense_ignore_value",
                    "defense_ignore_value_ungrouped",
                    "defense_ignore_value_empty",
                ]) and
                Check.not_none(groups, "defense_ignore_ratio")
            ), error_message="Sanity check failed", groups=groups)
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])

    def _inject_physical_line_2(self, context: MatchContext) -> None:
        source_text = context.get("physical_line_1")
        context.replace(text=source_text, subpattern_name="physical_line_2")


class MagicalDamageBuffInjector(DamageBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_magical_damage_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_first_value,
            self._inject_attack_first_ratio,
            self._inject_attack_final_value,
            self._get_attack_gain_pass(damage_type="magical"),
            self._get_damage_weakener_pass(damage_type="magical"),
            self._get_attack_final_ratio_pass(damage_type="magical"),
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "magical_main", "id", "attack_first_value", "attack_first_ratio",
            "attack_first_ratio_empty", "attack_final_value", "attack_multiplier",
            "attack_gain", "resist_loss_value", "resist_loss_value_ungrouped",
            "resist_loss_value_empty", "resist_loss_ratio", "resist_ignore_value",
            "resist_ignore_value_ungrouped", "resist_ignore_value_empty", 
            "resist_ignore_ratio", "resist_ungrouped", "damage_final_ratio",
            "damage_final_ratio_empty",
        ])
        Assert.not_none(groups, [
            "attack_first_value", "attack_final_value",
            "attack_multiplier", "attack_gain"
        ])
        Assert.not_empty(groups, ["magical_main", "id"])
        Assert.either(groups, ["attack_first_ratio", "attack_first_ratio_empty"])
        Assert.true(
            (
                Check.not_none(groups, "resist_loss_ratio") and
                Check.either(groups, [
                    "resist_loss_value",
                    "resist_loss_value_ungrouped",
                    "resist_loss_value_empty"
                ]) and
                Check.either(groups, [
                    "resist_ignore_value",
                    "resist_ignore_value_ungrouped",
                    "resist_ignore_value_empty",
                ]) and
                Check.not_none(groups, "resist_ignore_ratio")
            ) or (
                Check.not_none(groups, "resist_ungrouped")
            ), error_message="Sanity check failed", groups=groups)
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])


class TrueDamageBuffInjector(DamageBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_true_damage_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_first_value,
            self._inject_attack_first_ratio,
            self._inject_attack_final_value,
            self._get_attack_gain_pass(damage_type="true"),
            self._get_attack_final_ratio_pass(damage_type="true"),
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "true_main", "id", "attack_first_value", "attack_first_ratio",
            "attack_first_ratio_empty", "attack_final_value", "attack_multiplier",
            "attack_gain", "damage_final_ratio", "damage_final_ratio_empty",
        ])
        Assert.not_none(groups, [
            "attack_first_value", "attack_final_value",
            "attack_multiplier", "attack_gain"
        ])
        Assert.not_empty(groups, ["true_main", "id"])
        Assert.either(groups, ["attack_first_ratio", "attack_first_ratio_empty"])
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])


class ElementalDamageBuffInjector(DamageBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_elemental_damage_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_first_value,
            self._inject_attack_first_ratio,
            self._inject_attack_final_value,
            self._get_attack_gain_pass(damage_type="elemental"),
            self._get_attack_final_ratio_pass(damage_type="elemental"),
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "elemental_main", "id", "attack_first_value", "attack_first_ratio",
            "attack_first_ratio_empty", "attack_final_value", "attack_multiplier",
            "attack_gain", "damage_final_ratio", "damage_final_ratio_empty",
        ])
        Assert.not_none(groups, [
            "attack_first_value", "attack_final_value",
            "attack_multiplier", "attack_gain"
        ])
        Assert.not_empty(groups, ["elemental_main", "id"])
        Assert.either(groups, ["attack_first_ratio", "attack_first_ratio_empty"])
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])


class DefaultElementalDamageBuffInjector(DamageBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_default_elemental_damage_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._get_attack_final_ratio_pass(damage_type="elemental"),
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "default_elemental_main", "default_dark", "default_fire",
            "damage_final_ratio", "damage_final_ratio_empty",
        ])
        Assert.not_empty(groups, ["default_elemental_main"])
        Assert.either(groups, ["default_dark", "default_fire"])
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])


class InjuryBuffInjector(DamageBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_injury_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_first_value,
            self._inject_attack_first_ratio,
            self._inject_attack_final_value,
            self._inject_attack_gain_if_magical,
            self._get_damage_weakener_pass(damage_type="magical", allow_empty=True),
            self._get_attack_final_ratio_pass(damage_type="magical", allow_empty=True),
            self._inject_injury_final_ratio,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "injury_main", "injury_type", "id", "attack_first_value",
            "attack_first_ratio", "attack_first_ratio_empty", "attack_final_value",
            "attack_multiplier", "attack_gain", "resist_loss_value",
            "resist_loss_value_ungrouped", "resist_loss_value_empty",
            "resist_loss_ratio", "resist_ignore_value", "resist_ignore_value_ungrouped",
            "resist_ignore_value_empty", "resist_ignore_ratio", "resist_ungrouped",
            "damage_final_ratio", "damage_final_ratio_empty",
            "magical_injury_multiplier", "resist_empty", "injury_final_ratio",
            "injury_final_ratio_empty",
        ])
        Assert.not_none(groups, [
            "attack_first_value", "attack_final_value",
            "attack_multiplier", "attack_gain"
        ])
        Assert.not_empty(groups, ["injury_main", "injury_type", "id"])
        Assert.either(groups, ["attack_first_ratio", "attack_first_ratio_empty"])
        Assert.true(
            (
                (
                    (
                        Check.not_none(groups, "resist_loss_ratio") and
                        Check.either(groups, [
                            "resist_loss_value",
                            "resist_loss_value_ungrouped",
                            "resist_loss_value_empty"
                        ]) and
                        Check.either(groups, [
                            "resist_ignore_value",
                            "resist_ignore_value_ungrouped",
                            "resist_ignore_value_empty",
                        ]) and
                        Check.not_none(groups, "resist_ignore_ratio")
                    ) or (
                        Check.not_none(groups, "resist_ungrouped")
                    )
                ) and (
                    Check.either(groups, [
                        "damage_final_ratio",
                        "damage_final_ratio_empty",
                    ]) and
                    Check.not_none(groups, "magical_injury_multiplier")
                )
            ) or (
                Check.not_none(groups, "resist_empty")
            ), error_message="Sanity check failed", groups=groups)
        Assert.either(groups, ["injury_final_ratio", "injury_final_ratio_empty"])

    def _inject_attack_gain_if_magical(self, context: MatchContext) -> None:
        if context.match.group("resist_empty") is None:  # has magical resistance
            self._get_attack_gain_pass(damage_type="magical")(context)

    def _inject_injury_final_ratio(self, context: MatchContext) -> None:
        injury_buffs = {
            '@InjuryDark': '*BuffDamageInjuryFinalRatio*BuffDamageInjuryDarkFinalRatio',
            '@InjuryFire': '*BuffDamageInjuryFinalRatio*BuffDamageInjuryFireFinalRatio',
        }
        injury_type = context.match.group("injury_type")
        Assert.true(
            injury_type in injury_buffs, f"Unrecognized injury_type `{injury_type}`"
        )
        buff = injury_buffs[injury_type]
        if context.match.group("injury_final_ratio") is not None:
            context.insert(buff, after="injury_final_ratio")
        elif context.match.group("injury_final_ratio_empty") is not None:
            context.insert(f'*(1{buff})', after="injury_final_ratio_empty")
        else:
            Assert.never()


class AttackSpeedBuffInjector(BuffInjector):
    def __init__(self):
        super().__init__()

    def _get_attack_speed_buff(self, subject_not_none: Operator | Summon) -> str:
        buff = "+BuffDamageAttackSpeedFirstValue"
        if subject_not_none.is_operator() and subject_not_none.is_sniper():
            buff += "+BuffDamageAttackSpeedFirstValueSniper"
        if subject_not_none.is_melee():
            buff += "+BuffDamageAttackSpeedFirstValueMelee"
        elif subject_not_none.is_ranged():
            buff += "+BuffDamageAttackSpeedFirstValueRanged"
        else:
            Assert.never()
        return buff


class AttackSpanBuffInjector(AttackSpeedBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_attack_span_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_span_first_value,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "span", "span_ungrouped", "span_multiplier",
            "span_first_value", "span_first_value_empty",
        ])
        Assert.either(groups, ["span", "span_ungrouped"])
        Assert.not_none(groups, "span_multiplier")
        Assert.either(groups, ["span_first_value", "span_first_value_empty"])

    def _inject_span_first_value(self, context: MatchContext) -> None:
        context.ensure_subject_exist()
        buff = self._get_attack_speed_buff(context.subject)
        if context.match.group("span_first_value") is not None:
            context.insert(buff, after="span_first_value")
        else:
            context.insert(f'/((100{buff})/100)', after="span_first_value_empty")


class ManualAttackSpeedBuffInjector(AttackSpeedBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_manual_attack_speed_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_speed_first_value,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, ["manual_attack_speed_main", "attack_speed"])
        Assert.not_empty(groups, ["manual_attack_speed_main", "attack_speed"])

    def _inject_attack_speed_first_value(self, context: MatchContext) -> None:
        context.ensure_subject_exist()
        buff = self._get_attack_speed_buff(context.subject)
        context.insert(buff, after="attack_speed")


class AutomaticSkillBuffInjector(BuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_skill_automatic_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_automatic_skill_point_value,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "skill_automatic_main", "skill_automatic_max",
            "skill_automatic_recovery", "skill_automatic_recovery_empty",
        ])
        Assert.not_empty(groups, [
            "skill_automatic_main", "skill_automatic_max"
        ])
        Assert.either(groups, [
            "skill_automatic_recovery", "skill_automatic_recovery_empty"
        ])

    def _inject_automatic_skill_point_value(self, context: MatchContext) -> None:
        context.ensure_subject_exist()
        buff = '+BuffDamageSkillPointValueAutomatic'
        mono_buff = 'BuffDamageMonoSkillPointValueAutomatic'
        if context.subject.is_operator() and context.subject.is_caster():
            mono_buff += ',BuffDamageMonoSkillPointValueAutomaticCaster'
        elif context.subject.is_operator() and context.subject.is_supporter():
            mono_buff += ',BuffDamageMonoSkillPointValueAutomaticSupporter'
        if context.match.group("skill_automatic_recovery") is not None:
            draft_context = context.detach()
            self._inject_annotated_buffs(
                AnnotatedBuff("@MonoSkillPointAutomatic", mono_buff),
                subpattern_name="skill_automatic_recovery",
                context=draft_context,
            )
            draft_context.insert(buff, after="skill_automatic_recovery")
            recovery = draft_context.get("skill_automatic_recovery")
        elif context.match.group("skill_automatic_recovery_empty") is not None:
            recovery = f'+(N("@MonoSkillPointAutomatic")+MAX({mono_buff})){buff}'
        else:
            Assert.never()
        max_sp = context.get("skill_automatic_max")
        context.replace(
            f'((MATCH({max_sp}-0.00001,'
            f'INDEX(AutoSkill!$A$1:$U$3600,0,ROUND((0{recovery})/0.05,0)+1),1)+1)/30)',
            subpattern_name="skill_automatic_main",
        )


class OffensiveSkillAttackCountInjector(Transformer):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_skill_offensive_attack_count_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_offensive_attack_count_skill_point,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "skill_offensive_attack_main", "skill_offensive_attack_count", "attack_span",
        ])
        Assert.not_empty(groups, [
            "skill_offensive_attack_main", "skill_offensive_attack_count", "attack_span",
        ])

    def _inject_offensive_attack_count_skill_point(self, context: MatchContext) -> None:
        attack_span = context.match.group("attack_span")
        attack_count = context.match.group("skill_offensive_attack_count")
        context.replace(
            f'ROUNDDOWN(((MATCH({attack_count}-0.00001,'
            f'INDEX(AttackSkill!$A$1:$DA$1200,0,{attack_span}*30),1)+1)/30)'
            f'/{attack_span},0)',
            subpattern_name="skill_offensive_attack_main",
        )


class OffensiveSkillInjector(Transformer):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_skill_offensive_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_offensive_skill_point,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "skill_offensive_main", "skill_offensive_max", "attack_span",
        ])
        Assert.not_empty(groups, [
            "skill_offensive_main", "skill_offensive_max", "attack_span",
        ])

    def _inject_offensive_skill_point(self, context: MatchContext) -> None:
        attack_span = context.match.group("attack_span")
        max_sp = context.match.group("skill_offensive_max")
        context.replace(
            f'((MATCH({max_sp}-0.00001,'
            f'INDEX(AttackSkill!$A$1:$DA$1200,0,{attack_span}*30),1)+1)/30)',
            subpattern_name="skill_offensive_main",
        )


class EnduranceBuffInjector(BuffInjector):
    def __init__(self):
        super().__init__()

    def _inject_attack_final_value(self, context: MatchContext) -> None:
        buff = "-BuffCoverEnemyAttackLossValue"
        mono_buff = "BuffCoverMonoEnemyStolenAttackLossValue"
        joined_buffs = f'-(N("@MonoEnemyStolenAttack")+MAX({mono_buff})){buff}'
        if context.match.group("enemy_attack_loss_value") is not None:
            self._inject_annotated_buffs(
                AnnotatedBuff("@MonoEnemyStolenAttack", mono_buff),
                separator="-",
                subpattern_name="enemy_attack_loss_value",
                context=context,
            )
            context.insert(buff, after="enemy_attack_loss_value")
        else:
            context.insert(joined_buffs, after="enemy_attack_loss_value_empty")

    def _inject_attack_final_ratio(self, context: MatchContext) -> None:
        buff = "*BuffCoverEnemyAttackLossFinalRatio"
        mono_buff = "BuffCoverMonoEnemyWeakenedAttackLossRatio"
        if context.match.group("enemy_attack_loss_ratio") is not None:
            self._inject_annotated_buffs(
                AnnotatedBuff("@MonoEnemyWeakenedAttack", mono_buff, BuffJoinPolicy.MIN),
                separator="*",
                subpattern_name="enemy_attack_loss_ratio",
                context=context,
            )
            context.insert(buff, after="enemy_attack_loss_ratio")
        else:
            context.insert(
                f'*((N("@MonoEnemyWeakenedAttack")+MIN({mono_buff})){buff})',
                after="enemy_attack_loss_ratio_empty"
            )


class PhysicalEnduranceBuffInjector(EnduranceBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(_physical_endurance_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_final_value,
            self._inject_attack_final_ratio,
            self._inject_physical_line_2,
            self._inject_defense_first_value,
            self._inject_defense_first_ratio,
            self._inject_defense_final_value,
            self._inject_physical_final_ratio,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "physical_line_1", "enemy_attack_loss_value", "enemy_attack_loss_value_empty",
            "enemy_attack_loss_ratio", "enemy_attack_loss_ratio_empty", "physical_line_2",
            "id", "defense_first_value", "defense_first_ratio", "defense_first_ratio_empty",
            "defense_final_value", "damage_final_ratio", "damage_final_ratio_empty",
        ])
        Assert.either(groups, ["enemy_attack_loss_value", "enemy_attack_loss_value_empty"])
        Assert.either(groups, ["enemy_attack_loss_ratio", "enemy_attack_loss_ratio_empty"])
        Assert.not_none(groups, ["defense_first_value", "defense_final_value"])
        Assert.not_empty(groups, ["physical_line_1", "physical_line_2", "id"])
        Assert.either(groups, ["defense_first_ratio", "defense_first_ratio_empty"])
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])
        Assert.equal(groups, "physical_line_1", "physical_line_2")

    def _inject_physical_line_2(self, context: MatchContext) -> None:
        source_text = context.get("physical_line_1")
        context.replace(text=source_text, subpattern_name="physical_line_2")

    def _inject_defense_first_value(self, context: MatchContext) -> None:
        context.ensure_subject_exist()
        buff = "+BuffCoverDefenseFirstValue"
        if context.subject.is_melee():
            buff += "+BuffCoverDefenseFirstValueMelee"
        elif context.subject.is_ranged():
            buff += "+BuffCoverDefenseFirstValueRanged"
        context.insert(buff, after="defense_first_value")

    def _inject_defense_first_ratio(self, context: MatchContext) -> None:
        context.ensure_subject_exist()
        buff = "+BuffCoverDefenseFirstRatio"
        if context.subject.is_operator():
            if context.subject.is_defender():
                buff += "+BuffCoverDefenseFirstRatioDefender"
            elif context.subject.is_vanguard():
                buff += "+BuffCoverDefenseFirstRatioVanguard"
        if context.subject.is_melee():
            buff += "+BuffCoverDefenseFirstRatioMelee"
        elif context.subject.is_ranged():
            buff += "+BuffCoverDefenseFirstRatioRanged"
        if context.match.group("defense_first_ratio") is not None:
            context.insert(buff, after="defense_first_ratio")
        else:
            context.insert(f'*(1{buff})', after="defense_first_ratio_empty")

    def _inject_defense_final_value(self, context: MatchContext) -> None:
        buff = "+BuffCoverDefenseFinalValue"
        mono_buff = "BuffCoverMonoEncouragedDefenseFinalValue"
        self._inject_annotated_buffs(
            AnnotatedBuff("@MonoEncouragedDefense", mono_buff),
            subpattern_name="defense_final_value",
            context=context,
        )
        context.insert(buff, after="defense_final_value")

    def _inject_physical_final_ratio(self, context: MatchContext) -> None:
        mono_buff = "BuffCoverMonoShelterFinalRatio"
        evasion_buff = (
            "BuffCoverPhysicalEvasionFinalRatio*BuffCoverEnemyPhysicalHitRateLossFinalRatio"
        )
        damage_loss_buff = "BuffCoverPhysicalDamageLossFinalRatio"
        if context.match.group("damage_final_ratio") is not None:
            self._ensure_all_buffs_annotated("damage_final_ratio", context)
            self._inject_annotated_buffs(
                (
                    AnnotatedBuff("@MonoShelter", mono_buff, BuffJoinPolicy.MIN),
                    AnnotatedBuff("@Evasion", evasion_buff, BuffJoinPolicy.MULTIPLY),
                    AnnotatedBuff("@DamageLoss", damage_loss_buff, BuffJoinPolicy.MULTIPLY),
                ),
                separator="*",
                subpattern_name="damage_final_ratio",
                context=context,
            )
        else:
            context.insert(
                '*('
                f'(N("@MonoShelter")+MIN({mono_buff}))'
                f'*(N("@Evasion")+({evasion_buff}))'
                f'*(N("@DamageLoss")+({damage_loss_buff}))'
                ')',
                after="damage_final_ratio_empty"
            )


class MagicalEnduranceBuffInjector(EnduranceBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> None:
        return re.compile(_magical_endurance_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_final_value,
            self._inject_attack_final_ratio,
            self._inject_resist_first_value,
            self._inject_resist_first_ratio,
            self._inject_resist_final_value,
            self._inject_magical_final_ratio,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "enemy_attack_loss_value", "enemy_attack_loss_value_empty",
            "enemy_attack_loss_ratio", "enemy_attack_loss_ratio_empty",
            "id", "resist_first_value", "resist_first_ratio", "resist_first_ratio_empty",
            "resist_final_value", "damage_final_ratio", "damage_final_ratio_empty",
        ])
        Assert.either(groups, ["enemy_attack_loss_value", "enemy_attack_loss_value_empty"])
        Assert.either(groups, ["enemy_attack_loss_ratio", "enemy_attack_loss_ratio_empty"])
        Assert.not_empty(groups, "id")
        Assert.not_none(groups, ["resist_first_value", "resist_final_value"])
        Assert.either(groups, ["resist_first_ratio", "resist_first_ratio_empty"])
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])

    def _inject_resist_first_value(self, context: MatchContext) -> None:
        context.insert("+BuffCoverResistanceFirstValue", after="resist_first_value")

    def _inject_resist_first_ratio(self, context: MatchContext) -> None:
        buff = "+BuffCoverResistanceFirstRatio"
        if context.match.group("resist_first_ratio") is not None:
            context.insert(buff, after="resist_first_ratio")
        else:
            context.insert(f"*(1{buff})", after="resist_first_ratio_empty")

    def _inject_resist_final_value(self, context: MatchContext) -> None:
        context.insert("+BuffCoverResistanceFinalValue", after="resist_final_value")

    def _inject_magical_final_ratio(self, context: MatchContext) -> None:
        mono_buff = "BuffCoverMonoShelterFinalRatio"
        evasion_buff = (
            "BuffCoverMagicalEvasionFinalRatio*BuffCoverEnemyMagicalHitRateLossFinalRatio"
        )
        damage_loss_buff = "BuffCoverMagicalDamageLossFinalRatio"
        if context.match.group("damage_final_ratio") is not None:
            self._ensure_all_buffs_annotated("damage_final_ratio", context)
            self._inject_annotated_buffs(
                (
                    AnnotatedBuff("@MonoShelter", mono_buff, BuffJoinPolicy.MIN),
                    AnnotatedBuff("@Evasion", evasion_buff, BuffJoinPolicy.MULTIPLY),
                    AnnotatedBuff("@DamageLoss", damage_loss_buff, BuffJoinPolicy.MULTIPLY),
                ),
                separator="*",
                subpattern_name="damage_final_ratio",
                context=context,
            )
        else:
            context.insert(
                '*('
                f'(N("@MonoShelter")+MIN({mono_buff}))'
                f'*(N("@Evasion")+({evasion_buff}))'
                f'*(N("@DamageLoss")+({damage_loss_buff}))'
                ')',
                after="damage_final_ratio_empty"
            )


class TrueEnduranceBuffInjector(EnduranceBuffInjector):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> None:
        return re.compile(_true_endurance_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_final_value,
            self._inject_attack_final_ratio,
            self._inject_true_final_ratio,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "enemy_attack_loss_value", "enemy_attack_loss_value_empty",
            "enemy_attack_loss_ratio", "enemy_attack_loss_ratio_empty",
            "damage_final_ratio", "damage_final_ratio_empty",
        ])
        Assert.either(groups, ["enemy_attack_loss_value", "enemy_attack_loss_value_empty"])
        Assert.either(groups, ["enemy_attack_loss_ratio", "enemy_attack_loss_ratio_empty"])
        Assert.either(groups, ["damage_final_ratio", "damage_final_ratio_empty"])

    def _inject_true_final_ratio(self, context: MatchContext) -> None:
        evasion_buff = (
            "BuffCoverTrueEvasionFinalRatio*BuffCoverEnemyTrueHitRateLossFinalRatio"
        )
        damage_loss_buff = "BuffCoverTrueDamageLossFinalRatio"
        if context.match.group("damage_final_ratio"):
            self._ensure_all_buffs_annotated("damage_final_ratio", context)
            self._inject_annotated_buffs(
                (
                    AnnotatedBuff("@Evasion", evasion_buff, BuffJoinPolicy.MULTIPLY),
                    AnnotatedBuff("@DamageLoss", damage_loss_buff, BuffJoinPolicy.MULTIPLY),
                ),
                separator="*",
                subpattern_name="damage_final_ratio",
                context=context,
            )
        else:
            context.insert(
                '*('
                f'(N("@Evasion")+({evasion_buff}))'
                f'*(N("@DamageLoss")+({damage_loss_buff}))'
                ')',
                after="damage_final_ratio_empty"
            )


class HealthEnduranceBuffInjector(EnduranceBuffInjector):
    def __init__(self):
        super().__init__()

    def get_buffed_health(self, formula: str) -> str:
        return super()._run_and_extract_once(formula, "health_main")

    @Assert.override(Transformer)
    def _get_pattern(self) -> None:
        return re.compile(_endurance_health_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_health_first_value,
            self._inject_health_first_ratio,
            self._inject_health_final_value,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, [
            "health_main", "id", "health_first_value", "health_first_ratio",
            "health_first_ratio_empty", "health_final_value",
        ])
        Assert.not_empty(groups, ["health_main", "id"])
        Assert.not_none(groups, ["health_first_value", "health_final_value"])
        Assert.either(groups, ["health_first_ratio", "health_first_ratio_empty"])

    def _inject_health_first_value(self, context: MatchContext) -> None:
        context.insert("+BuffCoverHealthFirstValue", after="health_first_value")

    def _inject_health_first_ratio(self, context: MatchContext) -> None:
        context.ensure_subject_exist()
        buff = "+BuffCoverHealthFirstRatio"
        if context.subject.is_operator() and context.subject.is_defender():
            buff += "+BuffCoverHealthFirstRatioDefender"
        if context.subject.is_melee():
            buff += "+BuffCoverHealthFirstRatioMelee"
        elif context.subject.is_ranged():
            buff += "+BuffCoverHealthFirstRatioRanged"
        if context.match.group("health_first_ratio") is not None:
            context.insert(buff, after="health_first_ratio")
        else:
            context.insert(f"*(1{buff})", after="health_first_ratio_empty")

    def _inject_health_final_value(self, context: MatchContext) -> None:
        buff = "+BuffCoverHealthFinalValue"
        mono_buff = "BuffCoverMonoEncouragedHealthFinalValue"
        self._inject_annotated_buffs(
            AnnotatedBuff("@MonoEncouragedHealth", mono_buff),
            subpattern_name="health_final_value",
            context=context,
        )
        context.insert(buff, after="health_final_value")


class AttackSpeedLossEnduranceBuffInjector(EnduranceBuffInjector):
    def __init__(self):
        super().__init__()

    def exist(self, formula: str) -> bool:
        matches = re.finditer(self._get_pattern(), formula)
        for _ in matches:
            return True
        return False

    def get_buffed_attack_speed_loss(self, formula: str) -> str:
        return super()._run_and_extract_once(formula, "enemy_attack_speed_loss_value")

    @Assert.override(Transformer)
    def _get_pattern(self) -> None:
        return re.compile(_endurance_attack_speed_loss_regex, re.M | re.X)

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._inject_attack_speed_loss_value,
        ]

    @Assert.override(Transformer)
    def _ensure_valid(self, match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, "enemy_attack_speed_loss_value")
        Assert.not_none(groups, "enemy_attack_speed_loss_value")

    def _inject_attack_speed_loss_value(self, context: MatchContext) -> None:
        buff = "+BuffCoverEnemyAttackSpeedLossFirstValue"
        mono_buff_stolen = "BuffCoverMonoEnemyStolenAttackSpeedLossValue"
        mono_buff_cold = "BuffCoverMonoEnemyColdAttackSpeedLossValue"
        self._inject_annotated_buffs(
            (
                AnnotatedBuff("@MonoEnemyStolenAttackSpeed", mono_buff_stolen),
                AnnotatedBuff("@MonoEnemyColdAttackSpeed", mono_buff_cold),
            ),
            subpattern_name="enemy_attack_speed_loss_value",
            context=context,
        )
        context.insert(buff, after="enemy_attack_speed_loss_value")


class WholeTransformer(Transformer):
    def __init__(self) -> None:
        self._all_pattern = re.compile(r"(?P<all>.+)", re.M | re.X | re.S)
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return self._all_pattern

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, "all")
        Assert.not_empty(groups, "all")
        Assert.true(match.group("all").startswith("="), "Expect formula to start with `=`")


class EnduranceFormulaGenerator(WholeTransformer):
    def __init__(self, transform_mode: TransformMode):
        self._transform_mode = transform_mode
        self._health_processor = HealthEnduranceBuffInjector()
        self._attack_speed_processor = AttackSpeedLossEnduranceBuffInjector()
        self._evasion_pattern = r'\(\s*N\(\s*"@Evasion"\s*\)\+\(\s*[a-zA-Z0-9.*]+\s*\)\s*\)'
        self._damage_loss_pattern = r'\(\s*N\(\s*"@DamageLoss"\s*\)\+\(\s*[a-zA-Z0-9.*]+\s*\)\s*\)'
        self._shelter_pattern = r'\(\s*N\(\s*"@MonoShelter"\s*\)\+MIN\(\s*[a-zA-Z0-9.,]+\s*\)\s*\)'
        self._projectile_removal_pattern = (
            r'\(\s*N\(\s*"@ProjectileRemoval"\s*\)\+IF\(\s*EnemyRanged,\s*[0-9.]+,\s*1\s*\)\s*\)'
        )
        super().__init__()

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._generate,
        ]

    def _generate(self, context: MatchContext) -> None:
        inner_formula = context.match.group("all")[1:]
        endurance_without_evasion = self._remove_endurance_evasion_buff(inner_formula)
        endurance_without_damage_final_ratio = (
            self._remove_endurance_damage_final_ratio(inner_formula)
        )
        endurance = self._append_endurance_final_buff(inner_formula)
        health = self._get_health(context.formula)
        crystal_barrier = str(self._get_crystal_barrier(context.formula))
        if self._transform_mode.to_buff() or self._transform_mode.to_decay():
            self._generate_decay(
                context, endurance, endurance_without_evasion,
                endurance_without_damage_final_ratio, health, crystal_barrier,
            )
        elif self._transform_mode.to_enemy():
            attack_speed_loss = self._get_attack_speed_loss(context.formula)
            invalidate_condition = self._get_invalidate_condition(context)
            self._generate_enemy(
                context, endurance, endurance_without_evasion,
                endurance_without_damage_final_ratio, health, crystal_barrier,
                attack_speed_loss, invalidate_condition,
            )
        else:
            Assert.never()

    def _generate_decay(
            self,
            context: MatchContext,
            endurance: str,
            endurance_without_evasion: str,
            endurance_without_damage_final_ratio: str,
            health: str,
            crystal_barrier: str,
        ) -> None:
        result = (
            self._decay_template
                .replace(
                    "${endurance_without_damage_final_ratio}",
                    endurance_without_damage_final_ratio
                )
                .replace("${endurance_without_evasion}", endurance_without_evasion)
                .replace("${endurance}", endurance)
                .replace("${health}", health)
                .replace("${crystal_barrier}", crystal_barrier)
        )
        context.replace(text=result, subpattern_name="all")

    def _generate_enemy(
            self,
            context: MatchContext,
            endurance: str,
            endurance_without_evasion: str,
            endurance_without_damage_final_ratio: str,
            health: str,
            crystal_barrier: str,
            attack_speed_loss: str,
            invalidate_condition: str,
        ) -> None:
        result = (
            self._enemy_template
                .replace("${invalidate_condition}", invalidate_condition)
                .replace(
                    "${endurance_without_damage_final_ratio}",
                    endurance_without_damage_final_ratio
                )
                .replace("${endurance_without_evasion}", endurance_without_evasion)
                .replace("${endurance}", endurance)
                .replace("${health}", health)
                .replace("${crystal_barrier}", crystal_barrier)
                .replace("${attack_speed_loss}", attack_speed_loss)
        )
        context.replace(text=result, subpattern_name="all")

    def _append_endurance_final_buff(self, inner_formula: str) -> str:
        return (
            f"(({inner_formula})*IF(EnemyRanged,BuffCoverProjectileRemovalRatio,1))"
        )

    def _remove_endurance_evasion_buff(self, inner_formula: str) -> str:
        for pattern in (
            self._evasion_pattern,
            self._projectile_removal_pattern
        ):
            inner_formula = re.sub(pattern=pattern, repl="1", string=inner_formula)
        return f"({inner_formula})"

    def _remove_endurance_damage_final_ratio(self, inner_formula: str) -> str:
        for pattern in (
            self._evasion_pattern,
            self._projectile_removal_pattern,
            self._damage_loss_pattern,
            self._shelter_pattern,
        ):
            inner_formula = re.sub(pattern=pattern, repl="1", string=inner_formula)
        return f"({inner_formula})"

    def _get_block_count(self, context: MatchContext) -> int:
        context.ensure_subject_exist()
        block_count: int | None = self._extract_block_count(context.formula)
        return block_count if block_count is not None else context.subject.block_count()

    def _extract_block_count(self, formula: str) -> int | None:
        return self._extract_optional_annotated_integer(formula, "@BlockCount")

    def _get_plain_position(self, context: MatchContext) -> PlainPosition:
        context.ensure_subject_exist()
        force_melee = self._exist_optional_annotation(context.formula, "@PositionMelee")
        force_ranged = self._exist_optional_annotation(context.formula, "@PositionRanged")
        Assert.true(
            not (force_melee and force_ranged), "Expect at most one position annotation"
        )
        if force_melee:
            return PlainPosition.MELEE
        elif force_ranged:
            return PlainPosition.RANGED
        else:
            if context.subject.is_ranged_strict():
                return PlainPosition.RANGED
            else:
                return PlainPosition.MELEE

    def _get_crystal_barrier(self, formula: str) -> int:
        crystal_barrier: int | None = self._extract_crystal_barrier(formula)
        return crystal_barrier if crystal_barrier is not None else 0

    def _extract_crystal_barrier(self, formula: str) -> int | None:
        return self._extract_optional_annotated_integer(formula, "@CrystalBarrier")

    def _extract_optional_annotated_integer(self, formula: str, annotation: str) -> int | None:
        pattern = rf'\+0\*\(\s*N\(\s*"{annotation}"\s*\)\+(?P<value>[0-9]+)\s*\)'
        matches = list(re.finditer(pattern, formula))
        count = len(matches)
        Assert.true(count <= 1, f"Expect 0 or 1 `{annotation}` in formula, got {count}")
        if count == 0:
            return None
        groups = matches[0].groupdict()
        Assert.is_number(groups, "value")
        return int(groups["value"])

    def _exist_optional_annotation(self, formula: str, annotation: str) -> bool:
        pattern = rf'N\(\s*"{annotation}"\s*\)'
        matches = list(re.finditer(pattern, formula))
        count = len(matches)
        Assert.true(count <= 1, f"Expect 0 or 1 `{annotation}` in formula, got {count}")
        return count == 1

    def _get_health(self, formula: str) -> str:
        return self._health_processor.get_buffed_health(formula)

    def _get_attack_speed_loss(self, formula: str) -> str:
        base_text = (
            formula
            if self._attack_speed_processor.exist(formula)
            else f'{formula}+0*(N("@EnemyAttackSpeedLoss")+0)'
        )
        attack_speed_loss = (
            self._attack_speed_processor.get_buffed_attack_speed_loss(base_text)
        )
        Assert.true(
            attack_speed_loss.startswith("+"),
            f"Illegal attack speed loss: {attack_speed_loss}"
        )
        return f"({attack_speed_loss[1:]})"

    def _get_invalidate_condition(self, context: MatchContext) -> str:
        if (
            self._get_plain_position(context) == PlainPosition.RANGED or
            self._get_block_count(context) == 0
        ):
            return "NOT(EnemyRanged)"
        else:
            return "FALSE"

    _decay_template = """
=IFS(
  ${endurance_without_damage_final_ratio}<=(BuffCoverCrystalBarrierValue+${crystal_barrier}),1,
  ${endurance_without_evasion}*BuffCoverDamageDelayRatio>=${health},NA(),
  TRUE,1-${endurance}/EnemyDamagePerHit
)
    """.strip()

    _enemy_template = """
=IFS(
  ${invalidate_condition},NA(),
  OR(EnemyDamagePerHit=0,EnemyAttackSpan=0),0,
  EnemyDamagePerHit/EnemyAttackSpan>1500,1/0,
  ${endurance_without_damage_final_ratio}<=(BuffCoverCrystalBarrierValue+${crystal_barrier}),0,
  ${endurance_without_evasion}*BuffCoverDamageDelayRatio>=${health},1500,
  TRUE,(${endurance}/(ROUND((EnemyAttackSpan/(MAX(100-${attack_speed_loss},20)/100))*30,0)/30))
)
    """.strip()


class BuffInvalidateWrapper(WholeTransformer):
    def __init__(self, transform_mode: TransformMode):
        self._invalid_value = self._get_invalid_value(transform_mode)
        prefix = self._get_buff_prefix(transform_mode)
        self._apply_to_single_ally_only_literal = f"{prefix}ApplyToSingleAllyOnly"
        self._apply_to_non_summon_ally_only_literal = f"{prefix}ApplyToNonSummonAllyOnly"
        super().__init__()

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._wrap_invalidate,
        ]

    def _wrap_invalidate(self, context: MatchContext) -> None:
        invalid = self._invalid_value
        formula = context.match.group("all")[1:]
        if not self._is_summon_buffable(context.summons):
            wrapped_formula = f'=IF(BuffSourceIds<>"",{invalid},{formula})'
        else:
            self_finders = ",".join(self._get_self_finders(context.subject_ids))
            if len(context.operators) == 1 and len(context.summons) == 0:
                wrapped_formula = (
                    f'=IF(OR('
                    f'{self_finders}'
                    f'),{invalid},{formula})'
                )
            elif len(context.operators) == 0 and len(context.summons) == 1:
                wrapped_formula = (
                    f'=IF(OR('
                    f'{self._apply_to_non_summon_ally_only_literal},'
                    f'{self_finders}'
                    f'),{invalid},{formula})'
                )
            elif len(context.operators) >= 1 and len(context.summons) >= 1:
                wrapped_formula = (
                    f'=IF(OR('
                    f'{self._apply_to_single_ally_only_literal},'
                    f'{self._apply_to_non_summon_ally_only_literal},'
                    f'{self_finders}'
                    f'),{invalid},{formula})'
                )
            elif len(context.operators) >= 2 and len(context.summons) == 0:
                wrapped_formula = (
                    f'=IF(OR('
                    f'{self._apply_to_single_ally_only_literal},'
                    f'{self_finders}'
                    f'),{invalid},{formula})'
                )
            else:
                Assert.never()
        context.replace(text=wrapped_formula, subpattern_name="all")

    def _is_summon_buffable(self, summons: Sequence[Summon]) -> bool:
        for summon in summons:
            if summon.has_multiple_positions():
                return False
        return True

    def _get_self_finders(self, subject_ids: Sequence[str]) -> Iterable[str]:
        short_subject_ids = (
            (
                subject_id[:-1]
                if subject_id[-1:] in ("D", "X", "Y", "Z")
                else subject_id
            )
            for subject_id
            in subject_ids
        )
        for short_subject_id in short_subject_ids:
            yield f'ISNUMBER(SEARCH("{short_subject_id}",BuffSourceIds))'

    def _get_invalid_value(self, transform_mode: TransformMode) -> str:
        if transform_mode.from_damage():
            return "0"
        elif transform_mode.from_endurance():
            return "NA()"
        else:
            Assert.never()

    def _get_buff_prefix(self, transform_mode: TransformMode) -> str:
        if transform_mode.from_damage():
            return "BuffDamage"
        elif transform_mode.from_endurance():
            return "BuffCover"
        else:
            Assert.never()


class SingleWordReplacer(Transformer):
    def __init__(self, keyword: str, replacement: str):
        self.keyword = keyword
        self.replacement = replacement
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(fr"(?P<keyword>\b{self.keyword}\b)", re.M | re.X)

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, ["keyword"])
        Assert.not_empty(groups, ["keyword"])

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            lambda context: context.replace(
                self.replacement, subpattern_name="keyword", allow_shorten=True
            )
        ]


class ReturnPredicateRemover(Transformer):
    def __init__(self):
        super().__init__()

    @Assert.override(Transformer)
    def _get_pattern(self) -> re.Pattern:
        return re.compile(r'Return[a-zA-Z]+,(?P<cell_predicate>C1="\w+"),', re.M | re.X)

    @Assert.override(Transformer)
    def _ensure_valid(self, match: re.Match) -> None:
        groups = match.groupdict()
        Assert.has_key(groups, ["cell_predicate"])
        Assert.not_empty(groups, ["cell_predicate"])

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            lambda context: context.replace(
                "FALSE", subpattern_name="cell_predicate", allow_shorten=True
            )
        ]


class Minifier(WholeTransformer):
    def __init__(self, minify_policy: MinifyPolicy):
        self._max_legal_length = 8192
        self._minify_policy = minify_policy
        super().__init__()

    @Assert.override(Transformer)
    def _get_passes(self) -> Iterable[Callable[[MatchContext], None]]:
        return [
            self._minify_according_to_policy,
        ]

    def _minify_according_to_policy(self, context: MatchContext) -> None:
        if not self._should_minify(context.formula):
            return
        context.replace(
            text=self._minify(context.formula),
            subpattern_name="all",
            allow_shorten=True
        )

    def _minify(self, formula: str) -> str:
        minimized_formula = self._minify_all(formula)
        if minimized_formula != formula:
            before_size = len(formula)
            after_size = len(minimized_formula)
            ratio = after_size / before_size
            print(f"👉 Minification: {before_size} -> {after_size} ({ratio:.0%})")
        return minimized_formula

    def _minify_all(self, formula: str) -> str:
        if not self._should_minify(formula):
            return formula
        minifiers = [
            self._minify_variables,
            self._minify_annotations,
            self._minify_whitespaces,
        ]
        for minify in minifiers:
            formula = minify(formula)
            if not self._should_minify(formula):
                return formula
        return formula

    def _minify_variables(self, formula: str) -> str:
        for original, short in self._variable_map.items():
            formula = re.sub(pattern=fr'\b{original}\b', repl=short, string=formula)
        for index, word in enumerate(self._auto_variables):
            formula = re.sub(pattern=fr'\b{word}\b', repl=f'_{index + 1}', string=formula)
        return formula

    def _minify_annotations(self, formula: str) -> str:
        return re.sub(pattern=r'N\("[^"]+?"\)', repl="0", string=formula)

    def _minify_whitespaces(self, formula: str) -> str:
        return re.sub(pattern=r"\s", repl="", string=formula)

    def _should_minify(self, formula: str) -> bool:
        if self._minify_policy == MinifyPolicy.NO_MINIFY:
            return False
        elif self._minify_policy == MinifyPolicy.MAX:
            return True
        elif self._minify_policy == MinifyPolicy.ON_DEMAND:
            return len(formula) > self._max_legal_length
        else:
            Assert.never()

    _variable_map = {
        "BuffDamageAttackFirstValue": "za",
        "BuffDamageAttackFinalValue": "zb",
        "BuffDamageAttackFirstRatio": "zc",
        "BuffDamageAttackFirstRatioVanguard": "zd",
        "BuffDamageAttackFirstRatioGuard": "ze",
        "BuffDamageAttackFirstRatioDefender": "zf",
        "BuffDamageAttackFirstRatioSniper": "zg",
        "BuffDamageAttackFirstRatioCaster": "zh",
        "BuffDamageAttackFirstRatioMedic": "zi",
        "BuffDamageAttackFirstRatioSupporter": "zj",
        "BuffDamageAttackFirstRatioSpecialist": "zk",
        "BuffDamageAttackFirstRatioMelee": "zl",
        "BuffDamageAttackSpeedFirstValue": "zm",
        "BuffDamageAttackSpeedFirstValueSniper": "zn",
        "BuffDamageAttackSpeedFirstValueMelee": "zo",
        "BuffDamageAttackSpeedFirstValueRanged": "zp",
        "BuffDamagePhysicalGainValue": "zq",
        "BuffDamageMagicalGainValue": "zr",
        "BuffDamageTrueGainValue": "zs",
        "BuffDamageElementalGainValue": "zt",
        "BuffDamagePhysicalFinalRatio": "zu",
        "BuffDamageMagicalFinalRatio": "zv",
        "BuffDamageTrueFinalRatio": "zw",
        "BuffDamageElementalFinalRatio": "zx",
        "BuffDamageInjuryFinalRatio": "zy",
        "BuffDamageInjuryFireFinalRatio": "zz",
        "BuffDamageInjuryDarkFinalRatio": "zaa",
        "BuffDamageSkillPointValueAutomatic": "zab",
        "BuffDamageSkillPointSupplyIntervals": "zac",
        "BuffDamageSkillPointSupplyIntervalsAutomatic": "zad",
        "BuffDamageSkillPointSupplyIntervalsOffensive": "zae",
        "BuffDamageSkillPointSupplyIntervalsDefensive": "zaf",
        "BuffDamageSkillPointSupplyGains": "zag",
        "BuffDamageSkillPointSupplyGainsAutomatic": "zah",
        "BuffDamageSkillPointSupplyGainsOffensive": "zai",
        "BuffDamageSkillPointSupplyGainsDefensive": "zaj",
        "BuffDamageSkillPointSupplyEnables": "zak",
        "BuffDamageSkillPointSupplyEnablesAutomatic": "zal",
        "BuffDamageSkillPointSupplyEnablesOffensive": "zam",
        "BuffDamageSkillPointSupplyEnablesDefensive": "zan",
        "BuffDamageSkillPointDefensiveGainWhenHit": "zao",
        "BuffDamageEnemyDefenseLossValue": "zap",
        "BuffDamageEnemyDefenseLossFinalRatio": "zaq",
        "BuffDamageEnemyResistanceLossValue": "zar",
        "BuffDamageEnemyResistanceLossFinalRatio": "zas",
        "BuffDamageMonoEncouragedAttackFinalValue": "zat",
        "BuffDamageMonoEnergizedAttackFirstRatio": "zau",
        "BuffDamageMonoEnergizedAttackFirstRatioMelee": "zav",
        "BuffDamageMonoEnemyStolenDefenseLossValue": "zaw",
        "BuffDamageMonoEnemyFrozenResistanceLossValue": "zax",
        "BuffDamageMonoEnemyVulnerableFinalRatio": "zay",
        "BuffDamageMonoEnemyVulnerablePhysicalFinalRatio": "zbh",
        "BuffDamageMonoEnemyVulnerableMagicalFinalRatio": "zaz",
        "BuffDamageMonoEnemyVulnerableElementalFinalRatio": "zba",
        "BuffDamageMonoEnemyVulnerableElementalFinalRatioFire": "zbf",
        "BuffDamageMonoEnemyVulnerableElementalFinalRatioDark": "zbg",
        "BuffDamageMonoSkillPointValueAutomatic": "zbc",
        "BuffDamageMonoSkillPointValueAutomaticCaster": "zbd",
        "BuffDamageMonoSkillPointValueAutomaticSupporter": "zbe",
        # last variable abbreviation: zbh

        "EnemyDefenseMajor": "zua",
        "EnemyDefenseMinor": "zub",
        "EnemyResistanceMajor": "zuc",
        "EnemyResistanceMinor": "zud",
        "EnemyElementalResistanceMajor": "zue",
        "EnemyElementalResistanceMinor": "zuf",
        "EnemyInjuryResistanceMajor": "zug",
        "EnemyInjuryResistanceMinor": "zuh",
        "EnemyRankMajor": "zui",
        "EnemyRankMinor": "zuj",
        "EnemyWeightMajor": "zuk",
        "EnemyWeightMinor": "zul",
        "EnemyDamageTimeWindow": "zum",
        "EnemyCountMinor": "zun",
        "EnemyAerial": "zuo",
        "EnemyMaxElementMajor": "zup",
        "EnemyMaxElementMinor": "zuq",

        "BaseAttackLT40X": "zxa",
        "BaseAttackRE03D": "zxb",
        "BaseAttackLN11D": "zxc",
        "BaseAttackNM06": "zxd",
    }

    _auto_variables = ()


_operator_id_regex = r"""
(?:
  BaseAttack
  |BaseHealth
  |BaseDefense
  |BaseResistance
)(?P<id>[a-zA-Z0-9]+)
"""

_summon_id_regex = r"""
(?:
  BaseSummonAttack
  |BaseSummonHealth
  |BaseSummonDefense
  |BaseSummonResistance
)(?P<id>[a-zA-Z0-9]+)
"""

_operator_or_summon_id_regex = r"""
(?:
  BaseAttack|BaseSummonAttack
  |BaseHealth|BaseSummonHealth
  |BaseDefense|BaseSummonDefense
  |BaseResistance|BaseSummonResistance
)(?P<id>[a-zA-Z0-9]+)
"""

_annotation_regex = r"""
\(\s*N\(\s*"(?P<key>@[A-Za-z]+)"\s*\)\+(?P<value>[0-9.]+)\s*\)
"""

_annotations_regex = r"""
\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
([+\-*/]\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
"""

_annotation_key_regex = r"""
N\(\s*"(?P<key>@[A-Za-z]+)"\s*\)
"""

_physical_damage_regex = r"""
MAX\(\s*
  (?P<physical_line_1>
    \(\s*
      \(\s*
        \(\s*
          (?:BaseAttack|BaseSummonAttack)(?P<id>[a-zA-Z0-9]+)
          (?P<attack_first_value>(?:\+[0-9]+)*)
        \s*\)
        (?:
          \*\(\s*1(?P<attack_first_ratio>(?:
            [+*/][0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
          )*)\s*\)
          |
          (?P<attack_first_ratio_empty>)
        )
        (?P<attack_final_value>(?:
          \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
        )*)
      \s*\)
      (?P<attack_multiplier>(?:\*[0-9.]+)*)
      (?P<attack_gain>(?:\+[0-9.]+)*)
    \s*\)
  )\*[0-9.]+,\s*
  (?P<physical_line_2>
    \(\s*
      \(\s*
        \(\s*(?:BaseAttack|BaseSummonAttack)[a-zA-Z0-9]+(?:\+[0-9]+)*\s*\)
        (?:\*\(\s*1(?:[+*/][0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*\s*\)|)
        (?:\+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
      \s*\)
      (?:\*[0-9.]+)*
      (?:\+[0-9.]+)*
    \s*\)
  )-
  (?:
    MAX\(\s*
      \(\s*
        \(\s*
          (?:EnemyDefenseMajor|EnemyDefenseMinor)
          (?:
            -\(\s*(?P<defense_loss_value>
              (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
              (?:\+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
            )\s*\)
            |
            -(?P<defense_loss_value_ungrouped>[0-9.]+)
            |
            (?P<defense_loss_value_empty>)
          )
        \s*\)
        (?P<defense_loss_ratio>(?:\*[0-9.]+)*)
        (?:
          -\(\s*(?P<defense_ignore_value>(?:[0-9.]+)(?:\+[0-9.]+)*)\s*\)
          |
          -(?P<defense_ignore_value_ungrouped>[0-9.]+)
          |
          (?P<defense_ignore_value_empty>)
        )
      \s*\)
      (?P<defense_ignore_ratio>(?:\*[0-9.]+)*)
      ,\s*0
    \s*\)
    |
    (?P<defense_ungrouped>EnemyDefenseMajor|EnemyDefenseMinor)
  )
\s*\)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
)
"""

_magical_damage_regex = r"""
(?P<magical_main>
  \(\s*
    \(\s*
      \(\s*
        (?:BaseAttack|BaseSummonAttack)(?P<id>[a-zA-Z0-9]+)
        (?P<attack_first_value>(?:\+[0-9]+)*)
      \s*\)
      (?:
        \*\(\s*1(?P<attack_first_ratio>(?:
          [+*/][0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
        )*)\s*\)
        |
        (?P<attack_first_ratio_empty>)
      )
      (?P<attack_final_value>(?:
        \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
      )*)
    \s*\)
    (?P<attack_multiplier>(?:\*[0-9.]+)*)
    (?P<attack_gain>(?:\+[0-9.]+)*)
  \s*\)
)
\*\(\s*MEDIAN
  \(\s*100-
    (?:
      \(\s*
        \(\s*
          (?:EnemyResistanceMajor|EnemyResistanceMinor)
          (?:
            -\(\s*(?P<resist_loss_value>[0-9.]+(?:\+[0-9.]+)*)\s*\)
            |
            -(?P<resist_loss_value_ungrouped>[0-9.]+(?:\+[0-9.]+)*)
            |
            (?P<resist_loss_value_empty>)
          )
        \s*\)
        (?P<resist_loss_ratio>(?:\*[0-9.]+)*)
        (?:
          -\(\s*(?P<resist_ignore_value>[0-9.]+(?:\+[0-9.]+)*)\s*\)
          |
          -(?P<resist_ignore_value_ungrouped>[0-9.]+(?:\+[0-9.]+)*)
          |
          (?P<resist_ignore_value_empty>)
        )
      \s*\)
      (?P<resist_ignore_ratio>(?:\*[0-9.]+)*)
      |
      (?P<resist_ungrouped>EnemyResistanceMajor|EnemyResistanceMinor)
    )
    ,\s*5,\s*100
  \s*\)/100
\s*\)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
)
"""

_true_damage_regex = r"""
(?P<true_main>
  \(\s*
    \(\s*
      \(\s*
        N\(\s*"@True"\s*\)\+
        (?:BaseAttack|BaseSummonAttack)(?P<id>[a-zA-Z0-9]+)
        (?P<attack_first_value>(?:\+[0-9]+)*)
      \s*\)
      (?:
        \*\(\s*1(?P<attack_first_ratio>(?:
          [+*/][0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
        )*)\s*\)
        |
        (?P<attack_first_ratio_empty>)
      )
      (?P<attack_final_value>(?:
        \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
      )*)
    \s*\)
    (?P<attack_multiplier>(?:\*[0-9.]+)*)
    (?P<attack_gain>(?:\+[0-9.]+)*)
  \s*\)
)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
)
"""

_elemental_damage_regex = r"""
(?P<elemental_main>
  \(\s*
    \(\s*
      \(\s*
        (?:BaseAttack|BaseSummonAttack)(?P<id>[a-zA-Z0-9]+)
        (?P<attack_first_value>(?:\+[0-9]+)*)
      \s*\)
      (?:
        \*\(\s*1(?P<attack_first_ratio>(?:
          [+*/][0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
        )*)\s*\)
        |
        (?P<attack_first_ratio_empty>)
      )
      (?P<attack_final_value>(?:
        \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
      )*)
    \s*\)
    (?P<attack_multiplier>(?:\*[0-9.]+)*)
    (?P<attack_gain>(?:\+[0-9.]+)*)
  \s*\)
)
\*\(\s*MEDIAN
  \(\s*100-(?:EnemyElementalResistanceMajor|EnemyElementalResistanceMinor),\s*0,\s*100
  \s*\)/100
\s*\)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
)
"""

_default_elemental_damage_regex = r"""
(?P<default_elemental_main>
  \(\s*(?:
    (?P<default_dark>800)
    |
    (?P<default_fire>7000)
  )\s*\)
)
\*\(\s*MEDIAN
  \(\s*100-(?:EnemyElementalResistanceMajor|EnemyElementalResistanceMinor),\s*0,\s*100
  \s*\)/100
\s*\)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
)
"""

_injury_regex = r"""
(?P<injury_main>
  \(\s*
    \(\s*
      \(\s*
        N\(\s*"(?P<injury_type>@(?:InjuryDark|InjuryFire))"\s*\)\+
        (?:BaseAttack|BaseSummonAttack)(?P<id>[a-zA-Z0-9]+)
        (?P<attack_first_value>(?:\+[0-9]+)*)
      \s*\)
      (?:
        \*\(\s*1(?P<attack_first_ratio>(?:
          [+*/][0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
        )*)\s*\)
        |
        (?P<attack_first_ratio_empty>)
      )
      (?P<attack_final_value>(?:
        \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
      )*)
    \s*\)
    (?P<attack_multiplier>(?:\*[0-9.]+)*)
    (?P<attack_gain>(?:\+[0-9.]+)*)
  \s*\)
)
(?:
  \*\(\s*MEDIAN
    \(\s*100-
      (?:
        \(\s*
          \(\s*
            (?:EnemyResistanceMajor|EnemyResistanceMinor)
            (?:
              -\(\s*(?P<resist_loss_value>[0-9.]+(?:\+[0-9.]+)*)\s*\)
              |
              -(?P<resist_loss_value_ungrouped>[0-9.]+(?:\+[0-9.]+)*)
              |
              (?P<resist_loss_value_empty>)
            )
          \s*\)
          (?P<resist_loss_ratio>(?:\*[0-9.]+)*)
          (?:
            -\(\s*(?P<resist_ignore_value>[0-9.]+(?:\+[0-9.]+)*)\s*\)
            |
            -(?P<resist_ignore_value_ungrouped>[0-9.]+(?:\+[0-9.]+)*)
            |
            (?P<resist_ignore_value_empty>)
          )
        \s*\)
        (?P<resist_ignore_ratio>(?:\*[0-9.]+)*)
        |
        (?P<resist_ungrouped>EnemyResistanceMajor|EnemyResistanceMinor)
      )
      ,\s*5,\s*100
    \s*\)/100
  \s*\)
  (?:
    \*\(\s*
      (?P<damage_final_ratio>
        (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
        (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
      )
    \s*\)
    |
    (?P<damage_final_ratio_empty>)
  )
  (?P<magical_injury_multiplier>(?:\*[0-9.]+)*)
  |
  (?P<resist_empty>)
)
\*\(\s*MEDIAN
  \(\s*
    100-(?:EnemyInjuryResistanceMajor|EnemyInjuryResistanceMinor),\s*
    0,\s*
    100
  \s*\)/100
\s*\)
(?:
  \*\(\s*
    (?P<injury_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<injury_final_ratio_empty>)
)
"""

_attack_span_regex = r"""
\(\s*
  ROUND\(\s*
    (?:
      \(\s*(?P<span>[0-9.]+(?:[+-][0-9.]+)*)\s*\)
      |
      (?P<span_ungrouped>[0-9.]+)
    )
    (?P<span_multiplier>(?:\*[0-9.]+)*)
    (?:
      /\(\s*\(\s*(?P<span_first_value>100(?:[+\-*/][0-9]+)+)\s*\)/100\s*\)
      |
      (?P<span_first_value_empty>)
    )
    \*30,\s*0
  \s*\)/30
  (?:\+0\*(?:BaseAttack|BaseSummonAttack)[a-zA-Z0-9]+)?
\s*\)
"""

_manual_attack_speed_regex = r"""
(?P<manual_attack_speed_main>
  \(\s*
    N\(\s*"@AttackSpeed"\s*\)\+
    (?P<attack_speed>[0-9]+)
    (?:\+0\*(?:BaseAttack|BaseSummonAttack)[a-zA-Z0-9]+)?
  \s*\)
)
"""

_skill_automatic_regex = r"""
(?P<skill_automatic_main>
  \(\s*
    N\(\s*"@SkillAutomatic"\s*\)\+
    (?P<skill_automatic_max>[0-9]+)
    (?:
      /\(\s*1
        (?P<skill_automatic_recovery>
          (?:
            [+\-*/][0-9.]+
            |
            \+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
          )*
        )
      \s*\)
      |
      (?P<skill_automatic_recovery_empty>)
    )
    (?:\+0\*(?:BaseAttack|BaseSummonAttack)[a-zA-Z0-9]+)?
  \s*\)
)
"""

_skill_offensive_attack_count_regex = r"""
(?P<skill_offensive_attack_main>
  \(\s*
    N\(\s*"@SkillOffensiveAttackCount"\s*\)\+(?P<skill_offensive_attack_count>[0-9]+)\+0\*
    (?P<attack_span>\(\s*
      ROUND\(\s*
        (?:
          \(\s*[0-9.]+(?:[+-][0-9.]+)*\s*\)
          |
          [0-9.]+
        )
        (?:\*[0-9.]+)*
        (?:
          /\(\s*\(\s*100(?:[+\-*/][0-9]+)+\s*\)/100\s*\)
          |
        )
        \*30,\s*0
      \s*\)/30
      (?:\+0\*(?:BaseAttack|BaseSummonAttack)[a-zA-Z0-9]+)?
    \s*\))
  \s*\)
)
"""

_skill_offensive_regex = r"""
(?P<skill_offensive_main>
  \(\s*
    N\(\s*"@SkillOffensive"\s*\)\+(?P<skill_offensive_max>[0-9]+)\*
    (?P<attack_span>\(\s*
      ROUND\(\s*
        (?:
          \(\s*[0-9.]+(?:[+-][0-9.]+)*\s*\)
          |
          [0-9.]+
        )
        (?:\*[0-9.]+)*
        (?:
          /\(\s*\(\s*100(?:[+\-*/][0-9]+)+\s*\)/100\s*\)
          |
        )
        \*30,\s*0
      \s*\)/30
      (?:\+0\*(?:BaseAttack|BaseSummonAttack)[a-zA-Z0-9]+)?
    \s*\))
  \s*\)
)
"""

_physical_endurance_regex = r"""
EnemyDamageType=LiteralPhysical,\s*MAX
\(\s*
  (?P<physical_line_1>
    \(\s*
      EnemyDamagePerHit
      (?:
        -(?P<enemy_attack_loss_value>
          (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
          (?:-[0-9.]+|-\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
        )
        |
        (?P<enemy_attack_loss_value_empty>)
      )
    \s*\)
    (?:
      \*\(\s*
        (?P<enemy_attack_loss_ratio>
          (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
          (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
        )
      \s*\)
      |
      (?P<enemy_attack_loss_ratio_empty>)
    )
  )
  \*0\.05,\s*
  (?P<physical_line_2>
    \(\s*
      EnemyDamagePerHit
      (?:
        -(?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
        (?:-[0-9.]+|-\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
        |
      )
    \s*\)
    (?:
      \*\(\s*
        (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
        (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
      \s*\)
      |
    )
  )
  -
  \(\s*
    \(\s*
      (?:BaseDefense|BaseSummonDefense)(?P<id>[a-zA-Z0-9]+)
      (?P<defense_first_value>(?:\+[0-9]+)*)
    \s*\)
    (?:
      \*\(\s*1(?P<defense_first_ratio>(?:[+*/][0-9.]+)*)\s*\)
      |
      (?P<defense_first_ratio_empty>)
    )
    (?P<defense_final_value>(?:
      \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
    )*)
  \s*\)
\s*\)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
),
"""

_magical_endurance_regex = r"""
EnemyDamageType=LiteralMagical,\s*
\(\s*
  EnemyDamagePerHit
  (?:
    -(?P<enemy_attack_loss_value>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:-[0-9.]+|-\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
    |
    (?P<enemy_attack_loss_value_empty>)
  )
\s*\)
(?:
  \*\(\s*
    (?P<enemy_attack_loss_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<enemy_attack_loss_ratio_empty>)
)
\*\(\s*MEDIAN
  \(\s*100-
    \(\s*
      \(\s*
        (?:BaseResistance|BaseSummonResistance)(?P<id>[a-zA-Z0-9]+)
        (?P<resist_first_value>(?:\+[0-9]+)*)
      \s*\)
      (?:
        \*\(\s*1(?P<resist_first_ratio>(?:[+*/][0-9.]+)*)\s*\)
        |
        (?P<resist_first_ratio_empty>)
      )
      (?P<resist_final_value>(?:\+[0-9.]+)*)
    \s*\),\s*5,\s*100
  \s*\)/100
\s*\)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
),
"""

_true_endurance_regex = r"""
EnemyDamageType=LiteralTrue,\s*
\(\s*
  EnemyDamagePerHit
  (?:
    -(?P<enemy_attack_loss_value>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:-[0-9.]+|-\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
    |
    (?P<enemy_attack_loss_value_empty>)
  )
\s*\)
(?:
  \*\(\s*
    (?P<enemy_attack_loss_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<enemy_attack_loss_ratio_empty>)
)
(?:
  \*\(\s*
    (?P<damage_final_ratio>
      (?:[0-9.]+|\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))
      (?:\*[0-9.]+|\*\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\))*
    )
  \s*\)
  |
  (?P<damage_final_ratio_empty>)
)
\s*\)
"""

_endurance_health_regex = r"""
\+0\*
(?P<health_main>
  \(\s*
    \(\s*
      (?:BaseHealth|BaseSummonHealth)(?P<id>[a-zA-Z0-9]+)
      (?P<health_first_value>(?:\+[0-9]+)*)
    \s*\)
    (?:
      \*\(\s*1(?P<health_first_ratio>(?:[+*/][0-9.]+)*)\s*\)
      |
      (?P<health_first_ratio_empty>)
    )
    (?P<health_final_value>(?:
      \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
    )*)
  \s*\)
)
"""

_endurance_attack_speed_loss_regex = r"""
\+0\*
\(\s*
  N\(\s*"@EnemyAttackSpeedLoss"\s*\)
  (?P<enemy_attack_speed_loss_value>(?:
    \+[0-9.]+|\+\(\s*N\(\s*"@[A-Za-z]+"\s*\)\+[0-9.]+\s*\)
  )*)
\s*\)
"""

if __name__ == "__main__":
    main()
