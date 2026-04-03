from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class ParsedTraceback:
    exception_type: str
    message: str
    text: str


TRACEBACK_PATTERN = re.compile(
    r"(Traceback \(most recent call last\):[\s\S]+?\n(?P<exc>[A-Za-z_][A-Za-z0-9_]*): (?P<msg>[^\n]+))"
)


class TracebackParser:
    def parse(self, text: str) -> list[ParsedTraceback]:
        matches = list(TRACEBACK_PATTERN.finditer(text))
        parsed: list[ParsedTraceback] = []
        for match in matches:
            parsed.append(
                ParsedTraceback(
                    exception_type=match.group("exc"),
                    message=match.group("msg"),
                    text=match.group(0),
                )
            )
        return parsed

