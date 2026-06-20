import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    return df

def get_recommendations(user_skills: list, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:


    user_profile_str = ", ".join(user_skills)


    corpus = df["skills"].tolist() + [user_profile_str]

    vectorizer = TfidfVectorizer(token_pattern=r"[^,]+")

    tfidf_matrix = vectorizer.fit_transform(
        [text.strip() for text in corpus]
    )


    user_vector = tfidf_matrix[-1]   
    item_vectors = tfidf_matrix[:-1]      

    similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()


    results = df.copy()
    results["match_score"] = similarity_scores
    results["match_percent"] = (results["match_score"] * 100).round(1)


    results = results.sort_values(by="match_score", ascending=False)


    top_results = results.head(top_n)

    return top_results[["job_role", "skills", "match_percent"]]


def get_user_input() -> list:

    print("=" * 60)
    print("  TECH STACK RECOMMENDER — Project 3: AI Recommendation Logic")
    print("=" * 60)
    print("\nEnter at least 3 skills or interests you have or want to use.")
    print("(e.g. Python, Cloud Computing, Automation, SQL, Docker...)\n")

    skills = []
    while len(skills) < 3:
        raw = input(f"Skill #{len(skills) + 1}: ").strip()
        if raw:
            skills.append(raw)
        else:
            print("  -> Please enter a non-empty skill.")


    while True:
        more = input("Add another skill? (press Enter to skip, or type a skill): ").strip()
        if not more:
            break
        skills.append(more)

    return skills

def display_recommendations(recommendations: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  YOUR TOP CAREER MATCHES")
    print("=" * 60)

    if recommendations.empty or recommendations["match_percent"].max() == 0:
        print("\nNo strong matches found. Try entering more common tech skills.")
        return

    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        print(f"\n#{rank}  {row['job_role']}  —  {row['match_percent']}% match")
        print(f"     Key skills: {row['skills']}")

    print("\n" + "=" * 60)

def main():
    df = load_dataset("raw_skills.csv")
    user_skills = get_user_input()
    recommendations = get_recommendations(user_skills, df, top_n=3)
    display_recommendations(recommendations)


if __name__ == "__main__":
    main()
