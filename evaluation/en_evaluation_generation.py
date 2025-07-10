import csv, json
import os
import pandas as pd


def process_csv_data(file_path, output_file_path):
    df = pd.read_csv(file_path)
    # Assuming the CSV has columns 'text' and 'label'
    processed_data = []
    with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["source", "talker", "text"])
        writer.writeheader()
        for index, row in df.iterrows():
            dialogue_id = row["dialogue_id"]
            row_json = json.loads(row["turns"])
            speakers = row_json['speaker_role']
            utterances = row_json['utterance']
            zipped = zip(speakers, utterances)
            for speaker, utterance in zipped:
                row_dict = {"source": f"locomo_{dialogue_id}","talker": speaker, "text": utterance}
                writer.writerow(row_dict)
                processed_data.append(row_dict)
            # break
                
    return processed_data

def process_typed_json(file_path, output_file_path):
    samples = json.load(open(file_path,'r', encoding='utf-8'))
    out_samples = {}
    for data in samples:

        out_data = {'sample_id': data['sample_id']}
        if data['sample_id'] in out_samples:
            out_data['qa'] = out_samples[data['sample_id']]['qa'].copy()
        else:
            out_data['qa'] = data['qa'].copy()
        conversations = data['conversation']
        list_of_turns = []
        for k, v in conversations.items():
            if "session_" in k and isinstance(v, list):
                time_session_field = f"{k}_date_time"
                
                time_session = conversations[time_session_field]
                
                for turn in v:
                    list_of_turns.append({
                        "source": turn['dia_id'],
                        "time":time_session,
                        "talker": turn['speaker'],
                        "text": turn['text']
                    })
        print(len(list_of_turns))
        output_file_path_with_id = f"{output_file_path.rsplit('.', 1)[0]}_{data['sample_id']}.csv"
        with open(output_file_path_with_id, 'w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=["source", "time", "talker", "text"])
            writer.writeheader()
            writer.writerows(list_of_turns)
            
        output_file_path_for_qa = f"{output_file_path.rsplit('.', 1)[0]}_{data['sample_id']}_qa.json"
        with open(output_file_path_for_qa, 'w', newline='', encoding='utf-8') as output_file:
            json.dump(out_data['qa'], output_file, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    file_path = "evaluation/eval_dataset/locomo10.json"
    output_file_path = "evaluation/eval_dataset/locomo/locomo.csv"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    process_typed_json(file_path, output_file_path)
    print(f"Processed {file_path} and saved to {output_file_path}.")