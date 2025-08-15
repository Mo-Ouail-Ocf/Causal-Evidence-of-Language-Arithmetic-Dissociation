from model.model_wrapper_update import ModelWrapper


if __name__ == "__main__":
    model = ModelWrapper()
    texts = ["Hello, world!", "How are you?"]
    hidden_states = model.extract_hidden_states(texts)
    print(hidden_states[0].shape)

    language_texts = ["hello world", "this is a test"]
    arithmetic_texts = ["2+2=4", "5*3=15"]

    model.identify_arithmetic_neurons(language_texts, arithmetic_texts,)