from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

from findex4.index import Index
from findex4.tokenizer import tokenize


class QueryNode(ABC):


    @abstractmethod
    def evaluate(self, index: Index) -> set[int]:

        ...

    def __and__(self, other: QueryNode) -> QueryNode:
        return And(self, other)

    def __or__(self, other: QueryNode) -> QueryNode:
        return Or(self, other)

    def __invert__(self) -> QueryNode:
        return Not(self)


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    term: str

    def evaluate(self, index: Index) -> set[int]:
        normalized = list(tokenize(self.term))
        if not normalized:
            return set()
        clean_term = normalized[0]
        if clean_term in index:
            return {p.doc_id for p in index[clean_term]}
        return set()


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.terms:
            return set()


        norm_terms = []
        for t in self.terms:
            toks = list(tokenize(t))
            if toks:
                norm_terms.extend(toks)

        if not norm_terms:
            return set()


        if len(norm_terms) == 1:
            return Term(norm_terms[0]).evaluate(index)


        candidate_docs: set[int] | None = None
        for term in norm_terms:
            if term not in index:
                return set()
            term_docs = {p.doc_id for p in index[term]}
            if candidate_docs is None:
                candidate_docs = term_docs
            else:
                candidate_docs &= term_docs

        if not candidate_docs:
            return set()

        matching_docs: set[int] = set()


        for doc_id in candidate_docs:

            positions_per_term: list[tuple[int, ...]] = []
            for term in norm_terms:
                postings = index[term]
                for p in postings:
                    if p.doc_id == doc_id:
                        positions_per_term.append(p.positions)
                        break

            if not all(positions_per_term):
                continue


            first_positions = positions_per_term[0]
            found = False
            for start_pos in first_positions:
                match = True
                for idx, pos_list in enumerate(positions_per_term[1:], start=1):
                    if (start_pos + idx) not in pos_list:
                        match = False
                        break
                if match:
                    found = True
                    break

            if found:
                matching_docs.add(doc_id)

        return matching_docs


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    node: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        all_docs = set(index._doc_meta.keys())
        return all_docs - self.node.evaluate(index)




TOKEN_RE = re.compile(r'\"[^\"]+\"|\(|\)|AND|OR|NOT|&&|\|\||!|~|[^\s()"]+')


def tokenize_query(query_str: str) -> list[str]:
    return TOKEN_RE.findall(query_str)


class Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            return Term("")
        node = self.parse_or()
        return node

    def parse_or(self) -> QueryNode:
        left = self.parse_and()
        while True:
            tok = self.peek()
            if tok in ("OR", "||", "|"):
                self.consume()
                right = self.parse_and()
                left = Or(left, right)
            else:
                break
        return left

    def parse_and(self) -> QueryNode:
        left = self.parse_not()
        while True:
            tok = self.peek()
            if tok in ("AND", "&&", "&"):
                self.consume()
                right = self.parse_not()
                left = And(left, right)
            elif tok is not None and tok not in (")", "OR", "||", "|"):
                right = self.parse_not()
                left = And(left, right)
            else:
                break
        return left

    def parse_not(self) -> QueryNode:
        tok = self.peek()
        if tok in ("NOT", "!", "~"):
            self.consume()
            operand = self.parse_not()
            return Not(operand)
        return self.parse_primary()

    def parse_primary(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            return Term("")

        if tok == "(":
            self.consume()  # "("
            node = self.parse_or()
            if self.peek() == ")":
                self.consume()  # ")"
            return node

        if tok.startswith('"') and tok.endswith('"'):
            self.consume()
            phrase_text = tok[1:-1]
            raw_tokens = list(tokenize(phrase_text))
            return Phrase(tuple(raw_tokens))

        self.consume()
        return Term(tok)



def parse(query_str: str) -> QueryNode:

    tokens = tokenize_query(query_str)
    if not tokens:
        return Term("")
    return Parser(tokens).parse()
def extract_terms(node: QueryNode) -> list[str]:

    if isinstance(node, Term):
        return list(tokenize(node.term))
    elif isinstance(node, Phrase):
        res = []
        for t in node.terms:
            res.extend(list(tokenize(t)))
        return res
    elif isinstance(node, And) or isinstance(node, Or):
        return extract_terms(node.left) + extract_terms(node.right)
    elif isinstance(node, Not):
        return []
    elif hasattr(node, "left") and hasattr(node, "right"):
        return extract_terms(node.left) + extract_terms(node.right)
    return []