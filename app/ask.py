from app.generation.answer_generator import generate_answer

def main():
    print("Legal RAG — Local Ollama")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Enter your legal query: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            result = generate_answer(question)

            print("\nAnswer:\n")
            print(result["answer"])

            if result["sources"]:
                print("\nSources:\n")

                for source in result["sources"]:
                    print(
                        f"[{source['label']}] {source['title']}"
                    )
                    print(
                        f"    Court: {source['court']} | "
                        f"Date: {source['date']}"
                    )
                    print(
                        f"    Chunk: {source['chunk_index']}"
                    )

                    if source["url"]:
                        print(f"    URL: {source['url']}")

                    print()

        except Exception as exc:
            print(f"\nError generating answer: {exc}\n")


if __name__ == "__main__":
    main()
