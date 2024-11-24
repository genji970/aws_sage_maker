
def peft_config():
    # LoRA 설정
    lora_config = LoraConfig(
        r=16,  # Low-rank 업데이트 행렬 차원
        lora_alpha=16,  # 스케일링 팩터
        lora_dropout=0.1,  # 드롭아웃 비율
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # QLoRA가 적용될 대상 모듈
    )

def model_build():
    # 모델 및 토크나이저 로드
    model_name = "..."
    base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map="auto",  # GPU와 CPU를 자동 분배
                                                      # torch_dtype="auto",            # 자동으로 적절한 데이터 타입(FP32, FP16 등) 선택
                                                      offload_folder="./offload",  # 메모리가 부족할 경우 CPU로 데이터를 오프로드
                                                      offload_state_dict=True)  # 가중치도 필요 시 CPU로 오프로드

    # 기존 model freeze
    for param in base_model.parameters():
        param.requires_grad = False

    model = get_peft_model(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
