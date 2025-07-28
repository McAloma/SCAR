import sys, os, json, re, time, pickle
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
from tqdm import tqdm
from collections import defaultdict

from src.encoder.text_encoders import BERT_Encoder, RoBERTa_Encoder, GPT2_Encoder

# wiki sample num 18315148




def build_and_save_title_category_map(
    page_sql_path,
    categorylinks_sql_path,
    save_path="title_category_map.pkl",
    return_result=True
):
    # 1. 解析 page.sql，构建 title → page_id 映射
    print(f"Parsing page.sql: {page_sql_path}")
    title_to_id = {}
    insert_re_page = re.compile(r"\((\d+),\d+,'([^']+)'")
    with open(page_sql_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("INSERT INTO"):
                matches = insert_re_page.findall(line)
                for pid, title in matches:
                    title = title.replace('_', ' ')
                    title_to_id[title] = int(pid)
    print(f"Parsed {len(title_to_id)} title-to-id entries.")

    # 2. 解析 categorylinks.sql，构建 page_id → categories 映射
    print(f"Parsing categorylinks.sql: {categorylinks_sql_path}")
    id_to_categories = {}
    insert_re_cat = re.compile(r"\((\d+),'([^']+)'")
    with open(categorylinks_sql_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("INSERT INTO"):
                matches = insert_re_cat.findall(line)
                for pid, cat in matches:
                    pid = int(pid)
                    id_to_categories.setdefault(pid, []).append(cat.replace('_', ' '))
    print(f"Parsed {len(id_to_categories)} id-to-category entries.")

    # 3. 构建 title → categories 映射
    title_to_categories = {}
    for title, pid in title_to_id.items():
        if pid in id_to_categories:
            title_to_categories[title] = id_to_categories[pid]
    print(f"Matched {len(title_to_categories)} titles with categories.")

    # 4. 保存
    with open(save_path, "wb") as f:
        pickle.dump(title_to_categories, f)
    print(f"Saved title-to-category map to {save_path}.")

    if return_result:
        return title_to_categories





class Wikipedia_Dataloader():
    def __init__(self, encoder_type="bert", category_map_path=None):
        if encoder_type == "bert":
            self.encoder = BERT_Encoder()
            self.save_path = "./data/embeddings/wikipedia/bert"
        elif encoder_type == "roberta":
            self.encoder = RoBERTa_Encoder()
            self.save_path = "./data/embeddings/wikipedia/roberta"
        elif encoder_type == "gpt2":
            self.encoder = GPT2_Encoder()
            self.save_path = "./data/embeddings/wikipedia/gpt2"
        else:
            raise ValueError("Unsupported encoder type. Choose from 'bert', 'roberta', or 'gpt2'.")
        
        os.makedirs(self.save_path, exist_ok=True)

        save_json_path = "./data/wikipedia/title2categories.pkl"
        if not os.path.exists(save_json_path):
            self.category_map = title_cat_map = build_and_save_title_category_map(
                page_sql_path="data/wikipedia/enwiki-latest-page.sql",
                categorylinks_sql_path="data/wikipedia/enwiki-latest-categorylinks.sql",
                save_path=save_json_path
            )
        else:
            with open(save_json_path, "rb") as f:
                self.category_map = pickle.load(f)

    def get_categories(self, title):
        return self.category_map.get(title, [])

    def map_category_to_coarse(self, categories):
        CATEGORY_MAPPING = {
            # 军事
            "Military_Army": ["Army", "Infantry", "Cavalry", "Brigade", "Regiment"],
            "Military_Navy": ["Navy", "Fleet", "Ship", "Submarine"],
            "Military_Airforce": ["Air Force", "Squadron", "Pilot", "Aircraft"],
            "Military_War": ["War", "Battle", "Campaign", "Conflict"],

            # 科学
            "Science_Physics": ["Physics", "Quantum", "Relativity", "Thermodynamics"],
            "Science_Chemistry": ["Chemistry", "Organic Chemistry", "Inorganic Chemistry", "Biochemistry"],
            "Science_Biology": ["Biology", "Zoology", "Botany", "Genetics"],
            "Science_Mathematics": ["Mathematics", "Algebra", "Calculus", "Geometry"],
            "Science_Earth": ["Geology", "Geography", "Meteorology", "Climate", "Seismology"],

            # 政治
            "Politics_Government": ["Government", "Administration", "Ministry", "Parliament", "Constitution"],
            "Politics_Elections": ["Election", "Voting", "Campaign", "Political Party"],
            "Politics_Leaders": ["President", "Prime Minister", "Chancellor", "Monarch", "Leader"],

            # 体育
            "Sports_Football": ["Football", "Soccer", "FIFA", "Goalkeeper"],
            "Sports_Olympics": ["Olympics", "Summer Olympics", "Winter Olympics", "Medal"],
            "Sports_Basketball": ["Basketball", "NBA", "Dribble", "Slam Dunk"],
            "Sports_Tennis": ["Tennis", "Wimbledon", "Grand Slam", "Racquet"],

            # 历史
            "History_Ancient": ["Ancient", "Egypt", "Rome", "Greece", "Mesopotamia"],
            "History_Medieval": ["Medieval", "Feudalism", "Crusades", "Knights"],
            "History_Modern": ["Revolution", "Industrial", "World War", "Cold War"],
            
            # 文化
            "Culture_Language": ["Language", "Linguistics", "Grammar", "Dialects"],
            "Culture_Religion": ["Religion", "Christianity", "Islam", "Buddhism", "Hinduism"],
            "Culture_Art": ["Art", "Painting", "Sculpture", "Photography"],
            "Culture_Literature": ["Literature", "Novel", "Poetry", "Drama"],

            # 娱乐
            "Entertainment_Movies": ["Movie", "Film", "Actor", "Director", "Cinema"],
            "Entertainment_Music": ["Music", "Song", "Album", "Singer", "Concert"],
            "Entertainment_TV": ["Television", "TV", "Show", "Episode", "Series"],
            "Entertainment_Games": ["Game", "Video Game", "Esports", "Console", "RPG"],

            # 社会
            "Society_Education": ["Education", "School", "University", "Student", "Curriculum"],
            "Society_Ethics": ["Ethics", "Morality", "Philosophy", "Values"],
            "Society_Sociology": ["Society", "Inequality", "Demography", "Community"],
            "Society_Law": ["Law", "Court", "Justice", "Legal", "Crime", "Prison"],

            # 商业与经济
            "Economy_Business": ["Business", "Company", "Corporation", "Entrepreneur"],
            "Economy_Finance": ["Finance", "Bank", "Stock", "Investment", "Economics"],
            "Economy_Trade": ["Trade", "Import", "Export", "Tariff", "WTO"],

            # 医疗与健康
            "Health_Medicine": ["Medicine", "Doctor", "Disease", "Treatment", "Hospital"],
            "Health_Psychology": ["Mental Health", "Depression", "Therapy", "Psychology"],

            # 技术与计算机
            "Tech_Computer": ["Computer", "AI", "Machine Learning", "Programming", "Algorithm"],
            "Tech_Internet": ["Internet", "Website", "Social Media", "Browser", "Email"],
            "Tech_Engineering": ["Engineering", "Electrical", "Mechanical", "Robotics"],

            # 食品与烹饪
            "Food_Cuisine": ["Food", "Cuisine", "Dish", "Cooking", "Recipe"],
            "Food_Nutrition": ["Nutrition", "Diet", "Vitamins", "Calories"],

            # 环境与地球
            "Environment": ["Environment", "Pollution", "Ecology", "Climate Change", "Conservation"],

            # 交通与基础设施
            "Infrastructure_Transport": ["Transport", "Railway", "Road", "Airport", "Metro", "Bus"],

            # 哲学与心理学
            "Philosophy_Psychology": ["Philosophy", "Existentialism", "Cognitive", "Freud", "Behavior"],
        }
        fine_labels = set()
        for cat in categories:
            for fine_label, keywords in CATEGORY_MAPPING.items():
                if any(k.lower() in cat.lower() for k in keywords):
                    fine_labels.add(fine_label)
        return list(fine_labels) or ["Unknown"]

    def get_all_wiki_files(self, root_dir):
        wiki_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.startswith("wiki_"):
                    wiki_files.append(os.path.join(root, file))
        return wiki_files
    
    def extract_title(self, doc_line: str):
        match = re.search(r'title="([^"]+)"', doc_line)
        return match.group(1) if match else "Untitled"
    
    def parse_wiki_file(self, file_path):
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    title = data.get("title", "Untitled")
                    content = data.get("text", "").strip()
                    if content:
                        documents.append((title, content))
                except json.JSONDecodeError as e:
                    print(f"[Error] JSON parse error in file {file_path}: {e}")
        # print(f"[Info] Parsed {len(documents)} documents from {file_path}")
        return documents
    
    def save_to_json(self, data, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def process_wiki_files(self, file_paths, batch_size=5000):
        batch_titles = []
        batch_contents = []
        file_count = 0
        doc_count = 0
        label_counter = defaultdict(int)  # ✅ 新增标签计数器

        for path in tqdm(file_paths, desc="Processing files"):
            docs = self.parse_wiki_file(path)
            for title, content in docs:
                batch_titles.append(title)
                batch_contents.append(content)
                doc_count += 1

                if len(batch_titles) >= batch_size:
                    try:
                        embeddings = self.encoder.encode_batch(batch_contents)

                        batch_results = []
                        for t, e in zip(batch_titles, embeddings):
                            categories = self.get_categories(t)
                            coarse_labels = self.map_category_to_coarse(categories)
                            
                            # ✅ 统计标签
                            for label in coarse_labels:
                                label_counter[label] += 1

                            batch_results.append({
                                "title": t,
                                "label": coarse_labels,
                                "embedding": e.tolist()
                            })

                        file_count += 1
                        out_path = os.path.join(self.save_path, f"wiki_embeddings_{file_count:05d}.json")
                        self.save_to_json(batch_results, out_path)
                        # print(f"Saved batch #{file_count} ({len(batch_results)} items) to {out_path}")
                    except Exception as e:
                        print(f"[Error] Failed to encode batch of {len(batch_titles)} documents: {e}")

                    batch_titles = []
                    batch_contents = []

        # 处理最后一个不满 batch 的部分
        if batch_titles:
            try:
                embeddings = self.encoder.encode_batch(batch_contents)
                batch_results = []
                for t, e in zip(batch_titles, embeddings):
                    categories = self.get_categories(t)
                    coarse_labels = self.map_category_to_coarse(categories)

                    # ✅ 统计标签
                    for label in coarse_labels:
                        label_counter[label] += 1

                    batch_results.append({
                        "title": t,
                        "label": coarse_labels,
                        "embedding": e.tolist()
                    })
                file_count += 1
                out_path = os.path.join(self.save_path, f"wiki_embeddings_{file_count:05d}.json")
                self.save_to_json(batch_results, out_path)
                # print(f"Saved final batch #{file_count} ({len(batch_results)} items) to {out_path}")
            except Exception as e:
                print(f"[Error] Failed to encode final batch: {e}")

        print(f"Total {doc_count} documents encoded and saved in {file_count} files.")

        # ✅ 保存标签统计结果
        label_stat_path = os.path.join(self.save_path, "label_distribution.json")
        self.save_to_json(dict(label_counter), label_stat_path)
        print(f"[Info] Label distribution saved to {label_stat_path}")

    def process_and_save_wikipedia(self, data_dir, batch_size=128):
        file_paths = self.get_all_wiki_files(data_dir)
        print(f"Found {len(file_paths)} wiki files.")

        self.process_wiki_files(file_paths, batch_size=batch_size)




if __name__ == "__main__":
    data_path = "./data/wikipedia/extracted_wiki"
    encoder = "gpt2"
    dataloader = Wikipedia_Dataloader(encoder_type=encoder)
    dataloader.process_and_save_wikipedia(data_path)

    # for encoder in ["bert", "roberta", "gpt2"]:
    #     dataloader = Wikipedia_Dataloader(encoder_type=encoder)
    #     dataloader.process_and_save_wikipedia(data_path)
    