from twelvelabs import TwelveLabs, TextInputRequest

client = TwelveLabs(api_key="tlk_04TMKK92AXRHWH29B5ASA3P22BSV")

''''
response = client.embed.v_2.create(
    input_type="text",
    model_name="marengo3.0",
    text=TextInputRequest(
        input_text="Hi how are you doing today?",
    ),
)

print(f"Number of embeddings: {len(response.data)}")
for embedding_data in response.data:
    print(f"Embedding dimensions: {len(embedding_data.embedding)}")
    print(f"First 10 values: {embedding_data.embedding[:10]}")'''
from twelvelabs import TwelveLabs

