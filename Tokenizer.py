def replace_padding_with_ignore(labels, padding_value=..., ignore_value=-100):
    """
    라벨에서 지정된 패딩 값을 -100으로 변환합니다.
    """
    return [ignore_value if token == padding_value else token for token in labels]

def tokenizing():
    # input_ids, attention_mask, labels 생성
    tokenized_data = [
        {
            **tokenizer(
                item['instruction'] + item['data'],
                padding='max_length',
                truncation=True,
                max_length=max_label_length
            ),
            'labels': replace_padding_with_ignore(
                tokenizer(
                    item['label'],
                    padding='max_length',
                    truncation=True,
                    max_length=max_label_length
                )['input_ids']
            )
        }
        for item in train_data
    ]
