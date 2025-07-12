class VectorStoreSearchResult {
  final String content;
  final String docId;
  final int rank;
  final double score;

  VectorStoreSearchResult({
    required this.content,
    required this.docId,
    required this.rank,
    required this.score,
  });

  factory VectorStoreSearchResult.fromJson(Map<String, dynamic> json) {
    return VectorStoreSearchResult(
      content: json['content'],
      docId: json['doc_id'].toString(),
      rank: json['rank'] ?? 0,
      score: (json['score'] ?? json['distance'] ?? 0).toDouble(),
    );
  }
}
