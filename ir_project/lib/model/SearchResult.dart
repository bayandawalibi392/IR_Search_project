class SearchResult {
  final String content;
  final String docId;
  final int rank;
  final double score;

  SearchResult({
    required this.content,
    required this.docId,
    required this.rank,
    required this.score,
  });

  factory SearchResult.fromJson(Map<String, dynamic> json) {
    return SearchResult(
      content: json['content'],
      docId: json['doc_id'],
      rank: json['rank'],
      score: (json['score'] as num).toDouble(),
    );
  }
}
