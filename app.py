from flask import Flask, request, jsonify, render_template
import pickle
import requests

app = Flask(__name__)

print("📦 Loading processed data...")

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

movies['title_lower'] = movies['title'].str.lower()

print("✅ Data loaded successfully")

API_KEY = "------------------------------------"  # 🔴 replace this

# Poster fetch
def fetch_poster(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get('results') and data['results'][0].get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['results'][0]['poster_path']

    except:
        print("⚠ API error for:", movie_name)

    return "https://via.placeholder.com/150"


# Recommendation
def recommend(movie):
    print("\n🔍 Searching for:", movie)

    movie = movie.strip().lower()
    matches = movies[movies['title_lower'].str.contains(movie)]

    if matches.empty:
        print("❌ Movie not found")
        return ["Movie not found"], []

    movie_index = matches.index[0]
    print("🎯 Matched:", movies.iloc[movie_index].title)

    distances = similarity[movie_index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names = []
    posters = []

    print("📽 Recommendations:")

    for i in movie_list:
        title = movies.iloc[i[0]].title
        print("➡", title)

        names.append(title)
        posters.append(fetch_poster(title))

    return names, posters


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend_api():
    movie = request.json['movie']

    names, posters = recommend(movie)

    if names[0] == "Movie not found":
        return jsonify({
            "names": ["Movie not in dataset"],
            "posters": []
        })

    return jsonify({"names": names, "posters": posters})


@app.route('/latest')
def latest():
    try:
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()

        results = []
        for m in data.get('results', [])[:6]:
            poster = (
                "https://image.tmdb.org/t/p/w500" + m['poster_path']
                if m.get('poster_path')
                else "https://via.placeholder.com/150"
            )

            results.append({
                "title": m['title'],
                "poster": poster
            })

        return jsonify(results)

    except:
        return jsonify([])


if __name__ == '__main__':
    app.run(debug=True)
