import torch
import torch.nn.functional as F
import numpy as np
from model import CharRNN, CharLSTM

def generate(model, seed_characters, temperature, char2idx, idx2char, length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        char2idx: character to index mapping
        idx2char: index to character mapping
        length: length of the generated sequence

    Returns:
        samples: generated characters
    """
    model.eval()
    device = next(model.parameters()).device

    # Convert seed characters to indices
    input_indices = torch.tensor([char2idx[char] for char in seed_characters], dtype=torch.long).unsqueeze(0).to(device)

    # Initialize hidden state
    if isinstance(model, CharRNN):
        hidden = model.init_hidden(1).to(device)
    else:
        hidden = tuple(h.to(device) for h in model.init_hidden(1))

    generated_indices = []

    with torch.no_grad():
        for i in range(length):
            output, hidden = model(input_indices, hidden)
            output = output / temperature
            probabilities = F.softmax(output[-1], dim=-1).cpu().numpy()[0]
            predicted_index = np.random.choice(len(probabilities), p=probabilities)
            generated_indices.append(predicted_index)

            input_indices = torch.tensor([[predicted_index]], dtype=torch.long).to(device)

    generated_characters = ''.join([idx2char[idx] for idx in generated_indices])
    samples = seed_characters + generated_characters

    return samples


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Shakespeare('shakespeare_train.txt')

    input_size = len(dataset.char2idx)
    hidden_size = 128
    output_size = input_size
    num_layers = 1


    model_rnn = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    model_rnn.load_state_dict(torch.load('best_rnn.pth'))

    model_lstm = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    model_lstm.load_state_dict(torch.load('best_lstm.pth'))

    seed_characters_list = ['I', 'He', 'will', 'citizen', 'love']
    temperature = 0.8
    length = 100

    print("Vanilla RNN Generated Samples:")
    for seed in seed_characters_list:
        print('-'*30)
        generated_text = generate(model_rnn, seed, temperature, dataset.char2idx, dataset.idx2char,length)
        print(f'Seed: {seed}\n\nGenerated:\n {generated_text}')

    print("LSTM Generated Samples:")
    for seed in seed_characters_list:
        print('-'*30)
        generated_text = generate(model_lstm, seed, temperature, dataset.char2idx, dataset.idx2char,length)
        print(f'Seed: {seed}\n\nGenerated:\n {generated_text}')