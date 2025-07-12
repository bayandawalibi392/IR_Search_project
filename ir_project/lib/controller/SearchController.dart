import 'dart:convert';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;

import '../model/SearchResult.dart';
import '../model/VectorStore.dart';

class tfidfController extends GetxController {
  var isLoading = false.obs;
  var results = <SearchResult>[].obs;
  var selectedGroup = 'quora'.obs  ;
  var selectedMethod = 'tfidf'.obs;
  var selectedRepresentation = 'bert'.obs;
  var error = ''.obs;
  var queryText = ''.obs;
  var useInvertedIndex = false.obs;
  var vector_results = <VectorStoreSearchResult>[].obs;

  // Future<void> searchTfidf(String query, bool useInverted) async {
  //   results.clear();
  //   error.value = '';
  //   isLoading.value = true;
  //
  //   final url = Uri.parse(useInverted
  //       ? 'http://localhost:5013/search-tfidf-inverted'
  //       : 'http://localhost:5010/search-tfidf');
  //
  //   try {
  //     final response = await http.post(url,
  //         headers: {'Content-Type': 'application/json'},
  //         body: jsonEncode({'query': query}));
  //
  //     if (response.statusCode == 200) {
  //       var data = json.decode(response.body);
  //       var parsed = List<Map<String, dynamic>>.from(data['results']);
  //       results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
  //     } else {
  //       error.value = "❌ فشل الاتصال: ${response.statusCode}";
  //     }
  //
  //   } catch (e) {
  //     error.value = "❌ استثناء: $e";
  //   } finally {
  //     isLoading.value = false;
  //   }
  // }
  Future<void> searchTfidf(String query, bool useInverted, String group) async {
    results.clear();
    error.value = '';
    isLoading.value = true;

    Uri url;

    if (group == 'quora') {
      url = Uri.parse(useInverted
          ? 'http://localhost:5013/search-tfidf-inverted'
          : 'http://localhost:5010/search-tfidf');
    } else if (group == 'webis') {
      url = Uri.parse(useInverted
          ? 'http://localhost:5002/search-inverted'
          : 'http://localhost:5000/query/tfidf');
    } else {
      error.value = "⚠️ مجموعة غير مدعومة لهذا النوع.";
      isLoading.value = false;
      return;
    }

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': query}),
      );
      print("🌐 Response Status: ${response.statusCode}");
      print("🌐 Response Body: ${response.body}");
      if (response.statusCode == 200) {
        print("🌐 Response Status: ${response.statusCode}");
        print("🌐 Response Body: ${response.body}");
        var data = json.decode(response.body);
        var parsed = List<Map<String, dynamic>>.from(data['results']);
        results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
      } else {
        error.value = "❌ فشل الاتصال: ${response.statusCode}\nالرد: ${response.body}";
      }
    } catch (e) {
      error.value = "❌ استثناء: $e";
    } finally {
      isLoading.value = false;
    }
  }

  // Future<void> searchBert(String query, bool useInverted) async {
  //   results.clear();
  //   error.value = '';
  //   isLoading.value = true;
  //
  //   final url = Uri.parse(useInverted
  //       ? 'http://localhost:5014/search-bert-inverted'
  //       : 'http://localhost:5011/search-bert');
  //
  //   try {
  //     final response = await http.post(url,
  //         headers: {'Content-Type': 'application/json'},
  //         body: jsonEncode({'query': query}));
  //
  //     if (response.statusCode == 200) {
  //       var data = json.decode(response.body);
  //       var parsed = List<Map<String, dynamic>>.from(data['results']);
  //       results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
  //     } else {
  //       error.value = "❌ فشل الاتصال: ${response.statusCode}";
  //     }
  //   } catch (e) {
  //     error.value = "❌ استثناء: $e";
  //   } finally {
  //     isLoading.value = false;
  //   }
  // }
  Future<void> searchBert(String query, bool useInverted, String group) async {
    results.clear();
    error.value = '';
    isLoading.value = true;

    Uri url;

    if (group == 'quora') {
      url = Uri.parse(useInverted
          ? 'http://localhost:5014/search-bert-inverted'
          : 'http://localhost:5011/search-bert');
    } else if (group == 'webis') {
      url = Uri.parse(useInverted
          ? 'http://127.0.0.1:5006/query/bert-inv'
          : 'http://localhost:5001/query/bert');
    } else {
      error.value = "⚠️ مجموعة غير مدعومة لهذا النوع.";
      isLoading.value = false;
      return;
    }

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': query}),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        var parsed = List<Map<String, dynamic>>.from(data['results']);
        results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
      } else {
        error.value = "❌ فشل الاتصال: ${response.statusCode}";
      }
    } catch (e) {
      error.value = "❌ استثناء: $e";
    } finally {
      isLoading.value = false;
    }
  }


  // Future<void> searchHybrid(String query, bool useInverted) async {
  //   results.clear();
  //   error.value = '';
  //   isLoading.value = true;
  //
  //   final url = Uri.parse(useInverted
  //       ? 'http://localhost:5015/search-hybrid-inverted'
  //       : 'http://localhost:5012/search-hybrid-parallel');
  //
  //   try {
  //     final response = await http.post(url,
  //         headers: {'Content-Type': 'application/json'},
  //         body: jsonEncode({'query': query}));
  //
  //     if (response.statusCode == 200) {
  //       var data = json.decode(response.body);
  //       var parsed = List<Map<String, dynamic>>.from(data['results']);
  //       results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
  //     } else {
  //       error.value = "❌ فشل الاتصال: ${response.statusCode}";
  //     }
  //   } catch (e) {
  //     error.value = "❌ استثناء: $e";
  //   } finally {
  //     isLoading.value = false;
  //   }
  // }
  Future<void> searchHybrid(String query, bool useInverted, String group) async {
    results.clear();
    error.value = '';
    isLoading.value = true;

    Uri url;

    if (group == 'quora') {
      url = Uri.parse(useInverted
          ? 'http://localhost:5015/search-hybrid-inverted'
          : 'http://localhost:5012/search-hybrid-parallel');
    } else if (group == 'webis') {
      url = Uri.parse(useInverted
          ? 'http://127.0.0.1:5005/search-hybrid-indexed'
          : 'http://127.0.0.1:5003/search-hybrid-parallel');
    } else {
      error.value = "⚠️ مجموعة غير مدعومة لهذا النوع.";
      isLoading.value = false;
      return;
    }

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': query}),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        var parsed = List<Map<String, dynamic>>.from(data['results']);
        results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
      } else {
        error.value = "❌ فشل الاتصال: ${response.statusCode}";
      }
    } catch (e) {
      error.value = "❌ استثناء: $e";
    } finally {
      isLoading.value = false;
    }
  }

  // Future<void> searchBM25(String query, bool useInverted) async {
  //   results.clear();
  //   error.value = '';
  //   isLoading.value = true;
  //
  //   final url = Uri.parse(useInverted
  //       ? 'http://localhost:5016/search-bm25-inverted'
  //       : 'http://localhost:5017/search-bm25');
  //
  //   try {
  //     final response = await http.post(url,
  //         headers: {'Content-Type': 'application/json'},
  //         body: jsonEncode({'query': query}));
  //
  //     if (response.statusCode == 200) {
  //       var data = json.decode(response.body);
  //       var parsed = List<Map<String, dynamic>>.from(data['results']);
  //       results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
  //     } else {
  //       error.value = "❌ فشل الاتصال: ${response.statusCode}";
  //     }
  //   } catch (e) {
  //     error.value = "❌ استثناء: $e";
  //   } finally {
  //     isLoading.value = false;
  //   }
  // }
  Future<void> searchBM25(String query, bool useInverted, String group) async {
    results.clear();
    error.value = '';
    isLoading.value = true;

    Uri url;

    if (group == 'quora') {
      url = Uri.parse(useInverted
          ? 'http://localhost:5016/search-bm25-inverted'
          : 'http://localhost:5017/search-bm25');
    } else if (group == 'webis') {
      url = Uri.parse(useInverted
          ? 'http://localhost:5008/query/bm25-all'
          : 'http://localhost:5007/query/bm25');
    } else {
      error.value = "⚠️ مجموعة غير مدعومة لهذا النوع.";
      isLoading.value = false;
      return;
    }

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': query}),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        var parsed = List<Map<String, dynamic>>.from(data['results']);
        results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
      } else {
        error.value = "❌ فشل الاتصال: ${response.statusCode}";
      }
    } catch (e) {
      error.value = "❌ استثناء: $e";
    } finally {
      isLoading.value = false;
    }
  }

    Future<void> searchHybridRealEstate(String query, bool useInverted) async {
      results.clear();
      error.value = '';
      isLoading.value = true;

      // نتحقق إذا كان المستخدم فعّل الفهرسة، نوقف التابع مباشرة
      if (useInverted) {
        error.value = "⚠️ لا يوجد دعم للفهرسة العكسية لهذا النوع من البحث.";
        isLoading.value = false;
        return;
      }

      final url = Uri.parse('http://localhost:5018/search-hybrid');

      try {
        final response = await http.post(
          url,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'query': query}),
        );

        if (response.statusCode == 200) {
          var data = json.decode(response.body);
          var parsed = List<Map<String, dynamic>>.from(data['results']);
          results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
        } else {
          error.value = "❌ فشل الاتصال: ${response.statusCode}";
        }
      } catch (e) {
        error.value = "❌ استثناء: $e";
      } finally {
        isLoading.value = false;
      }
    }
  // Future<void> searchVectorStore(String query, bool useInverted) async {
  //   results.clear();
  //   error.value = '';
  //   isLoading.value = true;
  //
  //   final url = Uri.parse('http://localhost:5009/query/bert-faiss');
  //   if (useInverted) {
  //     error.value = "⚠️ لا يوجد دعم للفهرسة العكسية لهذا النوع من البحث.";
  //     isLoading.value = false;
  //     return;
  //   }
  //   try {
  //     final response = await http.post(
  //       url,
  //       headers: {'Content-Type': 'application/json'},
  //       body: jsonEncode({'query': query}),
  //     );
  //
  //     if (response.statusCode == 200) {
  //       var data = json.decode(response.body);
  //       var parsed = List<Map<String, dynamic>>.from(data['results']);
  //       vector_results.value = parsed.map((e) => VectorStoreSearchResult.fromJson(e)).toList();
  //     } else {
  //       error.value = "❌ فشل الاتصال: ${response.statusCode}";
  //     }
  //   } catch (e) {
  //     error.value = "❌ استثناء: $e";
  //   } finally {
  //     isLoading.value = false;
  //   }
  // }
  Future<void> searchVectorStore(String query, bool useInverted, String group) async {
    results.clear();
    vector_results.clear();
    error.value = '';
    isLoading.value = true;

    if (useInverted) {
      error.value = "⚠️ لا يوجد دعم للفهرسة العكسية لهذا النوع من البحث.";
      isLoading.value = false;
      return;
    }

    Uri url;

    if (group == 'quora') {
      url = Uri.parse('http://localhost:5009/query/bert-faiss');
    } else if (group == 'webis') {
      url = Uri.parse('http://localhost:5020/search-faiss');
    } else {
      error.value = "⚠️ المجموعة غير مدعومة.";
      isLoading.value = false;
      return;
    }

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': query}),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        var parsed = List<Map<String, dynamic>>.from(data['results']);
        vector_results.value = parsed.map((e) => VectorStoreSearchResult.fromJson(e)).toList();
      } else {
        error.value = "❌ فشل الاتصال: ${response.statusCode}";
      }
    } catch (e) {
      error.value = "❌ استثناء: $e";
    } finally {
      isLoading.value = false;
    }
  }

  // Future<void> searchHybridVector(String query, bool useInverted) async {
  //   results.clear();
  //   vector_results.clear();
  //   error.value = '';
  //   isLoading.value = true;
  //   if (useInverted) {
  //     error.value = "⚠️ لا يوجد دعم للفهرسة العكسية لهذا النوع من البحث.";
  //     isLoading.value = false;
  //     return;
  //   }
  //
  //   final url = Uri.parse('http://localhost:5019/search-hybrid-indexed');
  //
  //   try {
  //     final response = await http.post(
  //       url,
  //       headers: {'Content-Type': 'application/json'},
  //       body: jsonEncode({'query': query}),
  //     );
  //
  //     if (response.statusCode == 200) {
  //       var data = json.decode(response.body);
  //       var parsed = List<Map<String, dynamic>>.from(data['results']);
  //       results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
  //     } else {
  //       error.value = "❌ فشل الاتصال: ${response.statusCode}";
  //     }
  //   } catch (e) {
  //     error.value = "❌ استثناء: $e";
  //   } finally {
  //     isLoading.value = false;
  //   }
  // }
  Future<void> searchHybridVector(String query, bool useInverted, String group) async {
    results.clear();
    vector_results.clear();
    error.value = '';
    isLoading.value = true;

    // لا يتم استخدام الفهرسة في هذا النوع
    if (useInverted) {
      error.value = "⚠️ لا يوجد دعم للفهرسة العكسية لهذا النوع من البحث.";
      isLoading.value = false;
      return;
    }

    Uri url;
    if (group == 'quora') {
      url = Uri.parse('http://localhost:5019/search-hybrid-indexed');
    } else if (group == 'webis') {
      url = Uri.parse('http://localhost:5004/search-hybrid-indexed');
    } else {
      error.value = "⚠️ المجموعة غير مدعومة.";
      isLoading.value = false;
      return;
    }

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'query': query}),
      );

      if (response.statusCode == 200) {
        var data = json.decode(response.body);
        var parsed = List<Map<String, dynamic>>.from(data['results']);
        results.value = parsed.map((e) => SearchResult.fromJson(e)).toList();
      } else {
        error.value = "❌ فشل الاتصال: ${response.statusCode}";
      }
    } catch (e) {
      error.value = "❌ استثناء: $e";
    } finally {
      isLoading.value = false;
    }
  }



  Future<void> search({
      required String query,
      required String method,
      required String group,
      required bool isIndexed,
    }) async {
      if (group != 'quora') {
        error.value = "⚠️ هذه الطريقة غير مدعومة حاليًا إلا لمجموعة Quora فقط.";
        results.clear(); // تفريغ النتائج السابقة
        return;
      }
      switch (method) {
        case 'tfidf':
          await searchTfidf(query, isIndexed, group);
          break;
        case 'bert':
          await searchBert(query, isIndexed , group);
          break;
        case 'hybrid':
          await searchHybrid(query, isIndexed,group);
          break;
        case 'bm25':
          await searchBM25(query, isIndexed,group);
          break;
        case 'bm25hybrid':
          await searchHybridRealEstate(query, isIndexed);
          break;
        case 'vector store':
          await searchVectorStore(query, isIndexed,group); // ✅ هذه الإضافة
          break;
          //هاد معفن
        case 'hybrid_vector':
          await searchHybridVector(query, isIndexed,group); // ✅ هذه الإضافة
          break;
        default:
          error.value = "⚠️ طريقة البحث غير مدعومة: $method";
      }
    }
  }
