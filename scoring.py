import os
import json
import pandas as pd


def check_ans(prompt, pred, ans):
    # 추론값이 비어있으면 fail
    if pred:
        # 추론값이 입력보다 짧은 경우만
        pred = pred.split('Q:')[0]
        if len(prompt) > len(pred):
            target = pred.strip()[:10]

            if ans == 1 and 'A' in target:
                if 'B' not in target and 'C' not in target and 'D' not in target:
                    return True, target
            elif ans == 2 and 'B' in target:
                if 'A' not in target and 'C' not in target and 'D' not in target:
                    return True, target
            elif ans == 3 and 'C' in target:
                if 'A' not in target and 'B' not in target and 'D' not in target:
                    return True, target
            elif ans == 4 and 'D' in target:
                if 'A' not in target and 'B' not in target and 'C' not in target:
                    return True, target

            return False, "wrong"
        else:
            return False, "long"

    return False, "none"


if __name__ == "__main__":
    file_path = "./output/beomi_llama-2-ko-7b.xlsx"
    save_path = file_path.replace(".xlsx", "_score.xlsx")
    df = pd.read_excel(file_path)

    spans = []
    scores = []
    for idx, row in df.iterrows():
        prompt = row["Prompt"]
        pred = row["Pred_Answer"]
        ans = row["Real_Answer"]
        is_right, span = check_ans(prompt, pred, ans)
        scores.append(1 if is_right else 0)
        spans.append(span)

    df['span'] = spans
    df['score'] = scores
    df.to_excel(save_path, index=False)
    print(f'Saved to ... {save_path}')
    print(f'전체 데이터 개수: {len(df)}')
    print(f'맞춘 데이터 개수: {sum(scores)}')
    print(f'정답 비율: {sum(scores)/len(df)*100}')