def data_load():
    df_train = pd.read_csv(train_save_path)
    df_test = pd.read_csv(test_save_path)

    # 결측값 처리
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    # 결측값 -> 0 -> 문자열
    df_train = df_train.replace(0, "there is no input")
    df_test = df_test.replace(0, "there is no input")

def data_structure():
    input_column = ['instruction', 'input']
    label_column = ['output']

    # 각 행(row)에 대해 지정된 문장을 생성하고 리스트에 저장
    train_data = [
        {
            "instruction": "I want you to response data. keep in mind that if 'there is no input' sentence in prompt it means there is no input to consider.",
            "data": ",".join([f"{col} : {row[col]}" for col in df_train[input_column]]),
            "label": f"label : {row['output']}"
            }
        for _, row in df_train.iterrows()
    ]

    # 특수부호 제거
    train_data = [
        {key: value.replace("\n", "").replace("\\", "") if isinstance(value, str) else value
         for key, value in item.items()}
        for item in train_data
    ]

    test_data = [
        {
            "instruction": "I want you to response data. keep in mind that if 'there is no input' sentence in prompt it means there is no input to consider.",
            "data": ",".join([f"{col} : {row[col]}" for col in df_test[input_column]])
        }
        for _, row in df_test.iterrows()
    ]

    # 특수부호 제거
    test_data = [
        {key: value.replace("\n", "").replace("\\", "") if isinstance(value, str) else value
         for key, value in item.items()}
        for item in test_data
    ]

    # data , label 최대 길이
    max_data_length = max(len(item["data"]) for item in train_data)
    max_label_length = max(len(item["label"]) for item in train_data)

