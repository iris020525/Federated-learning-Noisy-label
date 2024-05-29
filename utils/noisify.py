import random


def noisify_label(true_label, num_classes=10, noise_type="symmetric"):
    if noise_type == "symmetric":#对称噪声
        label_lst = list(range(num_classes))
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]#函数会随机选择一个与真实标签不同的类别作为噪声标签，并返回该标签

    elif noise_type == "pairflip":#成对翻转噪声
        return (true_label - 1) % num_classes#函数会返回真实标签值减去 1 后对类别数量取模的结果作为噪声标签
