from app.chunking.chunker import chunk_text

def test_empty_text():
    assert chunk_text("") == []

def test_short_text():
    result = chunk_text("A short legal passage.")
    assert len(result) == 1
    assert result[0]["chunk_text"] == "A short legal passage."


def test_long_text():
    text = ("Legal judgment paragraph.\n" * 200)
    result = chunk_text(text)

    assert len(result) > 1
    assert result[0]["chunk_index"] == 0
    assert result[1]["chunk_index"] == 1

    for chunk in result:
        assert chunk["chunk_text"]
        assert text[chunk["char_start"]:chunk["char_end"]] == (
            chunk["chunk_text"]
        )


if __name__ == "__main__":
    test_empty_text()
    test_short_text()
    test_long_text()
    print("All chunker tests passed.")