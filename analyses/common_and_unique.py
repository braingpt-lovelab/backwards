import json 


def main():
    with open(forward_fpath, 'r') as f:
        forwards = json.load(f)

    with open(backward_fpath, 'r') as f:
        backwards = json.load(f)

    # Replace Ġ with space from each key if there is one.
    forwards_filtered_keys = set()
    print(f"Number of forwards keys: {len(forwards.keys())}")
    for k in forwards.keys():
        k = k.replace(space_token, '')
        forwards_filtered_keys.add(k.lower())

    backwards_filtered_keys = set()
    print(f"Number of backwards keys: {len(backwards.keys())}")
    for k in backwards.keys():
        if k.startswith('Ġ'):
            k = 'Ġ' + k[1:][::-1]
        else:
            k = k[::-1]
        k = k.replace(space_token, '')
        backwards_filtered_keys.add(k.lower())

    # Save as txt
    with open('data/forwards_filtered.txt', 'w') as f:
        # Sort keys by alphabetical order
        forwards_filtered_keys = sorted(list(forwards_filtered_keys))
        print(f"Number of forwards filtered keys: {len(forwards_filtered_keys)}")
        for k in forwards_filtered_keys:
            f.write(f"{k}\n")

    with open('data/backwards_filtered.txt', 'w') as f:
        # Sort keys by alphabetical order
        backwards_filtered_keys = sorted(list(backwards_filtered_keys))
        print(f"Number of backwards filtered keys: {len(backwards_filtered_keys)}")
        for k in backwards_filtered_keys:
            f.write(f"{k}\n")

    # Save as txt, common and unique keys in filtered version.
    forwards_filtered_keys = set(forwards_filtered_keys)
    backwards_filtered_keys = set(backwards_filtered_keys)
    filtered_common_keys = forwards_filtered_keys.intersection(backwards_filtered_keys)
    print(f"Number of common keys: {len(filtered_common_keys)}")

    with open('data/filtered_common.txt', 'w') as f:
        # Sort keys by alphabetical order
        filtered_common_keys = sorted(list(filtered_common_keys))
        for k in filtered_common_keys:
            f.write(f"{k}\n")


if __name__ == "__main__":
    forward_fpath = 'data/vocab_neuro.json'
    backward_fpath = 'data/vocab_neuro_backwards.json'
    space_token = "Ġ"

    main()