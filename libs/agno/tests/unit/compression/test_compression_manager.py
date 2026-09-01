import pytest

from agno.models.message import Message


@pytest.mark.asyncio
async def test_ashould_compress_below_token_limit():
    """Test async should_compress returns False when below token limit."""
    from agno.compression.manager import CompressionManager
    from agno.models.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o")
    messages = [Message(role="user", content="Hello")]

    cm = CompressionManager(compress_tool_results=True, compress_token_limit=1000)

    sync_result = cm.should_compress(messages, model=model)
    async_result = await cm.ashould_compress(messages, model=model)

    assert sync_result == async_result
    assert sync_result is False


@pytest.mark.asyncio
async def test_ashould_compress_above_token_limit():
    """Test async should_compress returns True when above token limit."""
    from agno.compression.manager import CompressionManager
    from agno.models.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o")
    messages = [Message(role="user", content="Hello " * 100)]

    cm = CompressionManager(compress_tool_results=True, compress_token_limit=10)

    sync_result = cm.should_compress(messages, model=model)
    async_result = await cm.ashould_compress(messages, model=model)

    assert sync_result == async_result
    assert sync_result is True


@pytest.mark.asyncio
async def test_ashould_compress_disabled():
    """Test async should_compress returns False when compression disabled."""
    from agno.compression.manager import CompressionManager
    from agno.models.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o")
    messages = [Message(role="user", content="Hello")]

    cm = CompressionManager(compress_tool_results=False)

    sync_result = cm.should_compress(messages, model=model)
    async_result = await cm.ashould_compress(messages, model=model)

    assert sync_result == async_result
    assert sync_result is False


def test_should_compress_below_token_limit():
    """Test sync should_compress returns False when below token limit."""
    from agno.compression.manager import CompressionManager
    from agno.models.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o")
    messages = [Message(role="user", content="Hello")]

    cm = CompressionManager(compress_tool_results=True, compress_token_limit=1000)
    result = cm.should_compress(messages, model=model)

    assert result is False


def test_should_compress_above_token_limit():
    """Test sync should_compress returns True when above token limit."""
    from agno.compression.manager import CompressionManager
    from agno.models.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o")
    messages = [Message(role="user", content="Hello " * 100)]

    cm = CompressionManager(compress_tool_results=True, compress_token_limit=10)
    result = cm.should_compress(messages, model=model)

    assert result is True


def test_should_compress_disabled():
    """Test sync should_compress returns False when compression disabled."""
    from agno.compression.manager import CompressionManager

    messages = [Message(role="user", content="Hello")]

    cm = CompressionManager(compress_tool_results=False)
    result = cm.should_compress(messages)

    assert result is False


def test_should_compress_default_count_limit():
    """Test that compress_tool_results_limit defaults to 3 when nothing is set."""
    from agno.compression.manager import CompressionManager

    cm = CompressionManager()
    assert cm.compress_tool_results_limit == 3

    cm_with_token = CompressionManager(compress_token_limit=1000)
    assert cm_with_token.compress_tool_results_limit is None

    cm_with_count = CompressionManager(compress_tool_results_limit=5)
    assert cm_with_count.compress_tool_results_limit == 5


def test_should_compress_count_based_below_limit():
    """Test should_compress with count-based limit below threshold."""
    from agno.compression.manager import CompressionManager

    messages = [
        Message(role="user", content="Hello"),
        Message(role="tool", content="Result 1", tool_name="test"),
    ]

    cm = CompressionManager(compress_tool_results=True, compress_tool_results_limit=5)
    result = cm.should_compress(messages)

    assert result is False


def test_should_compress_count_based_above_limit():
    """Test should_compress with count-based limit above threshold."""
    from agno.compression.manager import CompressionManager

    messages = [
        Message(role="user", content="Hello"),
        Message(role="tool", content="Result 1", tool_name="test1"),
        Message(role="tool", content="Result 2", tool_name="test2"),
        Message(role="tool", content="Result 3", tool_name="test3"),
    ]

    cm = CompressionManager(compress_tool_results=True, compress_tool_results_limit=2)
    result = cm.should_compress(messages)

    assert result is True


def test_should_compress_excludes_already_compressed():
    """Already compressed messages should not count toward the limit."""
    from agno.compression.manager import CompressionManager

    messages = [
        Message(role="user", content="Hello"),
        Message(role="tool", content="Result 1", tool_name="test1", compressed_content="compressed"),
        Message(role="tool", content="Result 2", tool_name="test2", compressed_content="compressed"),
        Message(role="tool", content="Result 3", tool_name="test3"),
    ]

    cm = CompressionManager(compress_tool_results=True, compress_tool_results_limit=2)
    result = cm.should_compress(messages)

    assert result is False


@pytest.mark.asyncio
async def test_ashould_compress_count_based_below_limit():
    """Test async should_compress with count-based limit below threshold."""
    from agno.compression.manager import CompressionManager

    messages = [
        Message(role="user", content="Hello"),
        Message(role="tool", content="Result 1", tool_name="test"),
    ]

    cm = CompressionManager(compress_tool_results=True, compress_tool_results_limit=5)

    sync_result = cm.should_compress(messages)
    async_result = await cm.ashould_compress(messages)

    assert sync_result == async_result
    assert sync_result is False


@pytest.mark.asyncio
async def test_ashould_compress_count_based_above_limit():
    """Test async should_compress with count-based limit above threshold."""
    from agno.compression.manager import CompressionManager

    messages = [
        Message(role="user", content="Hello"),
        Message(role="tool", content="Result 1", tool_name="test1"),
        Message(role="tool", content="Result 2", tool_name="test2"),
        Message(role="tool", content="Result 3", tool_name="test3"),
    ]

    cm = CompressionManager(compress_tool_results=True, compress_tool_results_limit=2)

    sync_result = cm.should_compress(messages)
    async_result = await cm.ashould_compress(messages)

    assert sync_result == async_result
    assert sync_result is True


def test_compress_accepts_empty_string_result(monkeypatch):
    """An empty string is a successful compression result and should not be retried."""
    from agno.compression.manager import CompressionManager

    messages = [Message(role="tool", content="Result", tool_name="test")]
    cm = CompressionManager()
    call_count = 0

    def return_empty_string(tool_result, run_metrics=None):
        nonlocal call_count
        call_count += 1
        return ""

    monkeypatch.setattr(cm, "_compress_tool_result", return_empty_string)

    cm.compress(messages)
    cm.compress(messages)

    assert messages[0].compressed_content == ""
    assert call_count == 1
    assert cm.stats["tool_results_compressed"] == 1
    assert cm.stats["compressed_size"] == 0


@pytest.mark.asyncio
async def test_acompress_accepts_empty_string_result(monkeypatch):
    """Async compression should also record an empty successful result exactly once."""
    from agno.compression.manager import CompressionManager

    messages = [Message(role="tool", content="Result", tool_name="test")]
    cm = CompressionManager()
    call_count = 0

    async def return_empty_string(tool_result, run_metrics=None):
        nonlocal call_count
        call_count += 1
        return ""

    monkeypatch.setattr(cm, "_acompress_tool_result", return_empty_string)

    await cm.acompress(messages)
    await cm.acompress(messages)

    assert messages[0].compressed_content == ""
    assert call_count == 1
    assert cm.stats["tool_results_compressed"] == 1
    assert cm.stats["compressed_size"] == 0
