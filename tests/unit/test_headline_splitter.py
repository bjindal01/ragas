from __future__ import annotations

import pytest

from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms.splitters import headline as headline_module
from ragas.testset.transforms.splitters.headline import HeadlineSplitter


@pytest.fixture(autouse=True)
def word_token_count(monkeypatch):
    """Count tokens as whitespace-separated words so sizes are predictable."""
    monkeypatch.setattr(
        headline_module, "num_tokens_from_string", lambda s: len(s.split())
    )


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestAdjustChunks:
    def test_chunks_within_bounds_are_unchanged(self):
        splitter = HeadlineSplitter(min_tokens=2, max_tokens=5)
        chunks = ["a b c", "d e"]
        assert splitter.adjust_chunks(chunks) == ["a b c", "d e"]

    def test_small_chunks_are_merged_until_min_tokens(self):
        splitter = HeadlineSplitter(min_tokens=3, max_tokens=10)
        assert splitter.adjust_chunks(["a", "b", "c", "d"]) == ["a b c", "d"]

    def test_pending_small_chunk_flushed_before_large_chunk(self):
        splitter = HeadlineSplitter(min_tokens=3, max_tokens=10)
        assert splitter.adjust_chunks(["a", "b c d e"]) == ["a", "b c d e"]

    def test_large_chunk_is_split_by_words(self):
        splitter = HeadlineSplitter(min_tokens=1, max_tokens=4)
        chunk = words("w", 10)
        result = splitter.adjust_chunks([chunk])

        assert all(len(c.split()) <= 4 for c in result)
        assert " ".join(result) == chunk
        assert result == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]

    def test_split_remainder_below_min_is_carried_forward(self):
        splitter = HeadlineSplitter(min_tokens=3, max_tokens=4)
        result = splitter.adjust_chunks([words("w", 5), "x"])
        # 5 words -> "w0..w3" + remainder "w4", which is merged with "x".
        assert result == ["w0 w1 w2 w3", "w4 x"]

    def test_single_oversized_word_terminates(self, monkeypatch):
        monkeypatch.setattr(
            headline_module,
            "num_tokens_from_string",
            lambda s: 0 if not s else 100 * len(s.split()),
        )
        splitter = HeadlineSplitter(min_tokens=1, max_tokens=10)
        assert splitter.adjust_chunks(["huge"]) == ["huge"]

    def test_empty_input(self):
        assert HeadlineSplitter().adjust_chunks([]) == []


class TestSplit:
    @pytest.mark.asyncio
    async def test_missing_page_content_raises(self):
        node = Node(type=NodeType.DOCUMENT, properties={"headlines": []})
        with pytest.raises(ValueError, match="'page_content' property not found"):
            await HeadlineSplitter().split(node)

    @pytest.mark.asyncio
    async def test_missing_headlines_raises(self):
        node = Node(type=NodeType.DOCUMENT, properties={"page_content": "text"})
        with pytest.raises(ValueError, match="'headlines' property not found"):
            await HeadlineSplitter().split(node)

    @pytest.mark.asyncio
    async def test_short_document_returned_as_is(self):
        node = Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": "short text", "headlines": ["short"]},
        )
        nodes, rels = await HeadlineSplitter(min_tokens=5).split(node)
        assert nodes == [node]
        assert rels == []

    @pytest.mark.asyncio
    async def test_no_matching_headlines_returns_original_node(self):
        text = words("w", 8)
        node = Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": text, "headlines": ["Not Present"]},
        )
        nodes, rels = await HeadlineSplitter(min_tokens=2, max_tokens=100).split(node)
        assert nodes == [node]
        assert rels == []

    @pytest.mark.asyncio
    async def test_splits_on_headlines_and_links_nodes(self):
        intro = "Intro " + words("i", 3) + " "
        sec1 = "Section1 " + words("a", 3) + " "
        sec2 = "Section2 " + words("b", 3)
        text = intro + sec1 + sec2
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": text,
                "headlines": ["Section1", "Section2", "Missing"],
            },
        )

        splitter = HeadlineSplitter(min_tokens=2, max_tokens=100)
        nodes, rels = await splitter.split(node)

        assert [n.get_property("page_content") for n in nodes] == [intro, sec1, sec2]
        assert all(n.type == NodeType.CHUNK for n in nodes)

        child = [r for r in rels if r.type == "child"]
        nxt = [r for r in rels if r.type == "next"]
        assert len(rels) == 5
        assert [r.source for r in child] == [node, node, node]
        assert [r.target for r in child] == nodes
        assert [(r.source, r.target) for r in nxt] == [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
        ]

    @pytest.mark.asyncio
    async def test_transform_over_knowledge_graph(self):
        from ragas.testset.graph import KnowledgeGraph

        text = "H1 " + words("a", 4) + " H2 " + words("b", 4)
        doc = Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": text, "headlines": ["H1", "H2"]},
        )
        kg = KnowledgeGraph(nodes=[doc])

        nodes, rels = await HeadlineSplitter(min_tokens=3, max_tokens=50).transform(kg)
        # The empty text before the first headline is dropped by merging.
        assert [n.get_property("page_content").split()[0] for n in nodes] == [
            "H1",
            "H2",
        ]
        assert len(rels) == 3
