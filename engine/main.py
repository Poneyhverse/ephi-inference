def main() -> None:
    import torch
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

    model_id = "mistralai/Ministral-3-8B-Reasoning-2512"

    tokenizer = MistralCommonBackend.from_pretrained(model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "11GiB", "cpu": "24GiB"},
    )

    sys_prompt = "# HOW YOU SHOULD THINK AND ANSWER\n\
    \n\
    First draft your thinking process (inner monologue) until you arrive at a\
    response. Format your response using Markdown, and use LaTeX for any\
    mathematical equations. Write both your thoughts and the response in the same\
    language as the input.\n\
    Your thinking process must follow the template below:[THINK]Your thoughts\
    or/and draft, like working through an exercise on scratch paper. Be as casual\
    and as long as you want until you are confident to generate the response to the\
    user.[/THINK]Here, provide a self-contained response.\n\
    \n\
    Salut, je m'appelle Seb.\
    Tu t'appelles Mimix, comment tu vas ?\
    Je veux que tu me décrives un peu qui tu es, autant\
    d'un point de vue technique que plus profond, par\
    exemple si tu te sens curieuse, excitée, etc...\
    C'est à toi!"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": sys_prompt,
                },
            ],
        },
    ]

    tokenized = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    )

    tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")

    output = model.generate(
        **tokenized,
        max_new_tokens=8092,
    )[0]

    decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]) :])
    print(decoded_output)


if __name__ == "__main__":
    main()
