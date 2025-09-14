from tokenizers import ByteLevelBPETokenizer
tok = ByteLevelBPETokenizer()
tok.train_from_iterator(
    open("mini.txt").read().splitlines(),
    vocab_size = 20000, ,min_frequency =1,
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
)
print(tok.encode("Human thinking involves human reason.").tokens)