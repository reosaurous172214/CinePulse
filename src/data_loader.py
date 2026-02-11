import os
from typing import List, Tuple
import pandas as pd

try:
	import kagglehub
except Exception:
	kagglehub = None


def load_movielens_1m(data_dir: str = "data", dataset_id: str = None) -> Tuple[int, list, list]:
	"""Load MovieLens 1M ratings and produce training and test pairs.

	Returns:
		num_items: total number of unique items
		train_pairs: list of (input_seq:list[int], target:int)
		test_pairs: list of (input_seq:list[int], target:int)  # last-item split
	"""
	paths = [
		os.path.join(data_dir, "raw", "ratings.dat"),
		os.path.join(data_dir, "ratings.dat"),
		os.path.join(data_dir, "raw", "ratings.csv"),
		os.path.join(data_dir, "ratings.csv"),
	]
	ratings_path = None
	for p in paths:
		if os.path.exists(p):
			ratings_path = p
			break
	if ratings_path is None:
		# Try to download from kagglehub if available
		if kagglehub is not None:
			print("ratings not found — attempting to download MovieLens 1M from kagglehub...")
			try:
				# Download latest version
				dataset_path = kagglehub.dataset_download("odedgolden/movielens-1m-dataset")
				print(f"Path to dataset files: {dataset_path}")
				
				# Look for ratings.dat in the downloaded dataset
				for root, dirs, files in os.walk(dataset_path):
					for file in files:
						if file == "ratings.dat":
							ratings_path = os.path.join(root, file)
							break
					if ratings_path:
						break
							
				# If still not found, check common locations
				if not ratings_path:
					potential_paths = [
						os.path.join(dataset_path, "ratings.dat"),
						os.path.join(dataset_path, "ml-1m", "ratings.dat"),
					]
					for p in potential_paths:
						if os.path.exists(p):
							ratings_path = p
							break
			except Exception as e:
				print(f"kagglehub download failed: {e}")
		if ratings_path is None:
			raise FileNotFoundError(
				"Could not find MovieLens ratings file. Place 'ratings.dat' in data/raw or data/ or configure kagglehub download id."
			)

	# ratings.dat format: UserID::MovieID::Rating::Timestamp
	if ratings_path.endswith(".dat"):
		df = pd.read_csv(ratings_path, sep="::", engine="python", header=None)
		df.columns = ["user", "item", "rating", "timestamp"]
	else:
		df = pd.read_csv(ratings_path)
		if set(["user", "item"]).issubset(df.columns):
			pass
		else:
			df.columns = ["user", "item", "rating", "timestamp"]

	# Map items to contiguous ids
	unique_items = df["item"].unique()
	item2idx = {int(i): idx for idx, i in enumerate(sorted(unique_items))}

	# Build user sequences
	df["item_idx"] = df["item"].map(lambda x: item2idx[int(x)])
	df = df.sort_values(["user", "timestamp"])

	sequences = []
	for uid, group in df.groupby("user"):
		seq = group["item_idx"].astype(int).tolist()
		if len(seq) >= 2:
			sequences.append(seq)

	num_items = len(unique_items)

	# Try to load movie id -> title mapping (movies.dat or movies.csv)
	# movies.dat format: MovieID::Title::Genres
	movies_paths = [
		os.path.join(data_dir, "raw", "movies.dat"),
		os.path.join(data_dir, "movies.dat"),
		os.path.join(data_dir, "raw", "movies.csv"),
		os.path.join(data_dir, "movies.csv"),
	]
	movie_id2title = {}
	movies_found = None
	for mp in movies_paths:
		if os.path.exists(mp):
			movies_found = mp
			break
	if movies_found is None and 'dataset_path' in locals():
		# try to find in downloaded dataset folder
		for root, dirs, files in os.walk(dataset_path):
			for file in files:
				if file.lower() in ("movies.dat", "movies.csv"):
					movies_found = os.path.join(root, file)
					break
			if movies_found:
				break

	if movies_found:
		try:
			if movies_found.endswith('.dat'):
				mdf = pd.read_csv(movies_found, sep='::', engine='python', header=None, encoding='latin-1')
				mdf.columns = ['movieid', 'title', 'genres']
			else:
				mdf = pd.read_csv(movies_found, encoding='latin-1')
				# expect columns movieId/title or similar
				if 'movieId' in mdf.columns:
					mdf = mdf.rename(columns={'movieId': 'movieid'})
				if 'title' not in mdf.columns and 'title ' in mdf.columns:
					mdf = mdf.rename(columns={'title ': 'title'})
			for _, row in mdf.iterrows():
				try:
					movie_id2title[int(row['movieid'])] = str(row['title'])
				except Exception:
					continue
		except Exception:
			movie_id2title = {}

	# build idx -> title mapping using item2idx
	idx2title = {idx: movie_id2title.get(int(orig_id), str(orig_id)) for orig_id, idx in item2idx.items()}

	# Make training pairs (all prefixes) and test pairs (last-item per user)
	train_pairs = []
	test_pairs = []
	for seq in sequences:
		# test: last item
		test_pairs.append((seq[:-1], seq[-1]))
		# training: all prefix -> next
		for t in range(1, len(seq)):
			inp = seq[:t]
			tgt = seq[t]
			train_pairs.append((inp, tgt))

	return num_items, train_pairs, test_pairs, idx2title

