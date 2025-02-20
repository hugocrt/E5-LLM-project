import torch
import tiktoken
from previous_labs import generate, text_to_token_ids, token_ids_to_text


MODEL_PATH = "ft-model.pth"


def load_model(model_path):
    """Charge le modèle GPT-2 fine-tuné"""
    ft_model = torch.load(model_path, map_location="cuda")

    if isinstance(ft_model, torch.nn.DataParallel):
        ft_model = ft_model.module

    device = torch.device("cuda")
    ft_model.to(device)
    return ft_model


def translate(text, model, tokenizer, max_length=100):
    """Génère la traduction du texte anglais en français"""
    input_text = (f"\n\n### Role: You are a translation assistant."
                  f"\n### Task: Translate the given English sentence into French."
                  f"\n### English Sentence:\n{text}"
                  f"\n\n### French Translation:\n")

    torch.manual_seed(123)

    idx = text_to_token_ids(input_text, tokenizer)
    idx = idx.to('cuda')

    token_ids = generate(
        model=model,
        max_new_tokens=max_length,
        idx=idx,
        context_size=1024,
        top_k=5,
        temperature=0.2,
        eos_id=50256
    )

    translated_text = token_ids_to_text(token_ids, tokenizer)
    return translated_text.split("\n### French Translation:\n")[-1].strip()


def setup():
    model = load_model(MODEL_PATH)
    tokenizer = tiktoken.get_encoding("gpt2")

    print("\n=== Mode interactif de traduction ===")
    print("Entrez un texte en anglais à traduire.")
    print("Tapez 'exit' ou 'quit' pour quitter.\n")

    return model, tokenizer


def interactive_mode():
    """
    Interface interactive pour traduire plusieurs phrases sans recharger le modèle
    commande à entrer pour utiliser notre IA : python translate.py
    taper exit ou quit pour fermer la "session"
    """
    # on charge modèle, tokenizer et on affiche les instructions
    ft_model, tokenizer = setup()

    while True:
        text = input(">>> ")

        if not text or text.isspace():
            continue

        if text.lower() in {"exit", "quit"}:
            print("Fermeture de la session...")
            break

        translation = translate(text, ft_model, tokenizer)
        print(f"Traduction : {translation}\n")


if __name__ == "__main__":
    interactive_mode()
