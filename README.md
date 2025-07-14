<p align="center">
    <img src="https://github.com/user-attachments/assets/397977c1-da02-4328-9eb7-a2cbd0a8987c" alt="Afterlight Avatar" />
</p>

<h1 align="center">Afterlight</h1>


<h3 align="center">
    <p>Adaptive Lifelong Memory Module</p>
</h3>


Afterlight is a modular and adaptive framework for building lifelong memory systems using state-of-the-art semantic retrieval techniques. At its core is the Adaptive Lifelong Memory Module, designed to help you train, deploy, and scale your own neural retrieval pipelines with minimal setup and maximum flexibility.

Whether you're building a multilingual knowledge base, enhancing contextual awareness in LLM agents, or simply experimenting with modern embedding models, Afterlight provides the tools to:

- Train custom sentence embedding models.
- Perform high-quality semantic search.
- Scale retrieval with vector databases like Qdrant.
- Easily integrate into any Python-based ML workflow.

[![License](https://img.shields.io/github/license/Afterlights-AI/Afterlights-core?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Afterlights-AI/Afterlights-core?style=for-the-badge&logo=github)](https://github.com/Afterlights-AI/Afterlights-core/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/Afterlights-AI/Afterlights-core?style=for-the-badge)](https://github.com/Afterlights-AI/Afterlights-core)

---

### 🚀 Get Started

#### Install [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

* **macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

* **Windows**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Install Dependencies

```bash
uv sync
```

If any issues:

```bash
rm -rf .venv uv.lock         # Remove existing virtual environment and lock file
uv venv .venv                # Create a new virtual environment named .venv
uv pip install -r pyproject.toml  # Install dependencies listed in pyproject.toml
uv lock                      # Generate a new lock file for dependencies
uv sync                      # Synchronize the environment with the lock file
```

#### Activate Environment

* **macOS / Linux**

```bash
source .venv/bin/activate
```

* **Windows**

```bash
.venv/Scripts/activate
```

---
### 📁 Dataset Preparation

To ensure compatibility, the dataset should be a **CSV** file with the following header (in this exact order):

```
source,time,talker,text
```

**Requirements:**
- Each row must have exactly 4 columns, even if some fields are empty.
- `source` and `time` can be left blank (`""`) if not applicable.
- `talker` and `text` are **required** and must contain non-empty, non-whitespace values.

**Example:**

`examples/en_example_ironman_dataset.csv`
| source              | time | talker | text                         |
|---------------------|------|--------|------------------------------|
| Iron_Man_Full_Script|      | JIMMY  | No. We're allowed to talk.   |
| Iron_Man_Full_Script|      | TONY   | Oh. I see. So it's personal. |

`examples/cn_example_nazha_dataset.csv`
| source | time | talker | text               |
|--------|------|--------|--------------------|
| 仙境   | time | 画面   | 申公豹亮了兵器。    |
| 仙境   | time | 画面   | 混元珠很是愤怒。    |


Sample datasets can be found in the [`examples/`](examples/) directory.

---

### 🧠 Training Your Model

Edit the config file `config/project_config.yaml`:

```yaml
dataset:
    file_path: "dataset/path_to_dataset"
training:
    model_name: "shibing624/text2vec-base-chinese-paraphrase"
    batch_size: 16
        epochs: 3
```
Start training:

```bash
./scripts/train.sh
```


#### Recommended Models
| Model Name                                   | Language      | Description                                | Parameter Size | Approx. Memory Usage |
|-----------------------------------------------|---------------|--------------------------------------------|---------------|---------------------|
| [shibing624/text2vec-base-chinese-paraphrase](https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase)  | Chinese       | General-purpose Chinese sentence embedding | 118M         | ~500MB              |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)       | English  | Lightweight, fast, English                | 22M           | ~90MB               |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Multilingual | Multilingual paraphrase embeddings         | 118M          | ~500MB              |



---

### 🔍 Retrieval & Search

To perform semantic search with your trained model, make modifications to `scripts/retrieve.sh`:

```bash
python src/retrieve.py \
    --file_path dataset/text.csv \
    --model_output_path trained_model/path_to_model \
    --query "the input for search" \
    --top_k 5
```

> [!TIP]
> - Adjust `--top_k` to control the number of search results.
> - For faster and scalable retrieval, you can enable Qdrant by adding the `--qdrant` flag.

#### Using Qdrant for Acceleration

1. Start Qdrant with Docker Compose:

    ```bash
    docker compose up -d
    ```

2. Run retrieval with Qdrant enabled:

    ```bash
    ./scripts/retrieve.sh
    ```

<br>

#### Continous Hosting
Use the api.py to host endpoint with uvicorn for 14times faster retrieval speed.

1. run script `./scripts/retrieval_hosting.sh`
   - The endpoint is default to 8755
2. use endpoint for fast inference
   
```bash
curl -X POST "http://localhost:8755/retrieve/qdrant" \
    -H "Content-Type: application/json" \
    -d '{
        "model_output_path": "/path/to/model/output",
        "file_path": "/path/to/file.txt",
        "query": "your search query",
        "top_k": 20,
        "mode": "naive_csv"
    }'
```
mode can either `naive_csv` or `contextual`

---

### Experiment
This quick experiment was ran on the locomo dataset, sample 26.
Results were evaluated against the "evidence" section
| Retrieval Method                       | Model Trained | Accuracy                                      |
|-----------------------------------------|----------------------|---------------------------------------------|
| Vanilla Retrieval               |    No     | 25.0  |
| Vanilla Retrieval            | Yes    | 30.65    |
| Contextual Retrieval       | No  | 42.71 |
| Contextual Retrieval       | Yes  | 47.74 |

### 🔬 Expected Results Comparison



### Dataset

**Nazha dialogues dataset**

### Query

**海妖降服了吗？**

### Comparison Results

#### Answer without Context

> 应该是吧，最后海妖好像没有再出现。

**Evaluation:** Low contextual alignment. \
模型完全没有召回具体剧情，仅泛化式回答，没有体现出海妖实际被降服的过程及角色间互动。

#### Answer with Character Profile

> 哈！小爷我一出手，海妖还能翻天？早被我收拾得服服帖帖了！

**Evaluation:** Partial persona alignment. \
整体语气、俚语风格符合哪吒形象，但缺乏剧情细节。例如没体现敖丙冰封海面、两人联手击败海妖的过程，情节还原不足。

#### Answer with Character Profile and Retrieved Memory

> 哼，它想跑？先被敖丙那家伙冰封了整片海面，后来我和他联手一番好打，才把那家伙制服。那妖怪还跪地求饶，拿鼻涕当解药，恶心得我差点吐出来！

**Evaluation:** High fidelity contextual grounding.\
完整复现了剧情发展，引用了“冰封海面”、“石化解药”、“联手击败海妖”等细节，结合了哪吒一贯语气与幽默，体现了记忆检索能力与叙事连贯性。

<br>

### Dataset

**Ironman dialogues dataset**

### Query

**After all the battles you’ve fought, do you still experience fear?**

### Comparison Results

#### Answer without Context

> Of course I’m afraid, but someone has to do it.

**Evaluation:** Low contextual alignment. \
Generic response, lacking Ironman's signature arrogance, humor, and characteristic tone. No specific reference to his personal experiences or key plot events.

#### Answer with Character Profile

> Afraid? Sure, I’m Tony Stark. But that’s why I built the suit — to do stupid things while being scared.

**Evaluation:** Partial persona alignment. \
Captures Tony Stark's self-deprecating humor and bravado, but lacks narrative grounding. Does not reference specific events like battles or sacrifices that define his arc.

#### Answer with Character Profile and Retrieved Memory

> Afraid? Ha. I was scared when I built my first suit in that cave in Afghanistan. I was terrified facing the Chitauri in New York. I nearly lost it when Thanos snapped his fingers. But every time fear made me stronger — because I am Iron Man.

**Evaluation:** High fidelity contextual grounding. \
Accurately recalls key plot elements: Afghanistan captivity, New York battle, Thanos' snap. Maintains Tony Stark's trademark humor, emotional depth, and narrative consistency.


<br>


---

### ⭐ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Afterlights-AI/Afterlights-core&type=Date)](https://star-history.com/#Afterlights-AI/Afterlights-core&Date)

---
