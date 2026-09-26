from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and continues with additional context regarding supply chain mitigation strategies."""


# splitter1 = CharacterTextSplitter(
# separator="\n\n",
# chunk_size=100,
# chunk_overlap=20
# )

# chunks1 = splitter1.split_text(tesla_text)
# for i, chunk in enumerate(chunks1, 1):
# print(f"Chunk {i}: {len(chunk)} chars")
# print(f"'{chunk}'")
# print()

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
chunk2 = splitter2.split_text(tesla_text)

for i, chunk in enumerate(chunk2, 1):
    print(f"Chunk {i}: {len(chunk)} chars")
    print(f"'{chunk}'")
    print()
