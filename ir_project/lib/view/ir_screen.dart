// import 'package:flutter/material.dart';
// import 'package:get/get.dart';
//
// import '../controller/SearchController.dart';
//
// class SearchScreen extends StatelessWidget {
//   final tfidfController controller = Get.put(tfidfController());
//
//   final TextEditingController textController = TextEditingController();
//
//   void _handleSearch() {
//     final group = controller.selectedGroup.value;
//     final method = controller.selectedMethod.value;
//     final query = controller.queryText.value;
//
//     if (group == 'quora' && method == 'bert') {
//       controller.searchbert(query);
//     } else if (group == 'quora' && method == 'hybrid') {
//       controller.searchHybrid(query);
//     }else if (group == 'quora' && method == 'bm25') {
//       controller.searchBM25(query);
//     } else {
//       controller.searchtfidf();
//     }
//   }
//
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: const Text("🔍 IR Search Engine")),
//       body: Padding(
//         padding: const EdgeInsets.all(16.0),
//         child: Column(
//           children: [
//             // Dropdowns
//             Row(
//               children: [
//                 Expanded(
//                   child: Obx(() => DropdownButton<String>(
//                     value: controller.selectedGroup.value,
//                     onChanged: (val) => controller.selectedGroup.value = val!,
//                     items: ['quora', 'webis']
//                         .map((e) => DropdownMenuItem(value: e, child: Text(e)))
//                         .toList(),
//                   )),
//                 ),
//                 const SizedBox(width: 10),
//                 Expanded(
//                   child: Obx(() => DropdownButton<String>(
//                     value: controller.selectedMethod.value,
//                     onChanged: (val) => controller.selectedMethod.value = val!,
//                     items: ['tfidf', 'bert', 'hybrid', 'bm25', 'vector store']
//                         .map((e) => DropdownMenuItem(value: e, child: Text(e)))
//                         .toList(),
//                   )),
//                 ),
//               ],
//             ),
//             const SizedBox(height: 16),
//
//             // Query Input
//             TextField(
//               controller: textController,
//               decoration: const InputDecoration(
//                 labelText: "أدخل الاستعلام",
//                 border: OutlineInputBorder(),
//               ),
//               onChanged: (val) => controller.queryText.value = val,
//               onSubmitted: (_) => _handleSearch(), // <-- ✅ تم التعديل هنا
//             ),
//             const SizedBox(height: 16),
//
//             // Search Button
//             ElevatedButton.icon(
//               onPressed: _handleSearch, // <-- ✅ تم التعديل هنا
//               icon: const Icon(Icons.search),
//               label: const Text("ابحث الآن"),
//             ),
//             const SizedBox(height: 16),
//
//             // نتائج البحث
//             Expanded(
//               child: Obx(() {
//                 if (controller.isLoading.value) {
//                   return const Center(child: CircularProgressIndicator());
//                 }
//                 if (controller.results.isEmpty) {
//                   return const Center(child: Text("لا توجد نتائج"));
//                 }
//                 return ListView.builder(
//                   itemCount: controller.results.length,
//                   itemBuilder: (context, index) {
//                     final item = controller.results[index];
//                     return Card(
//                       elevation: 3,
//                       margin: const EdgeInsets.symmetric(vertical: 6),
//                       child: ListTile(
//                         leading: CircleAvatar(child: Text("${item.rank}")),
//                         title: Text(item.content),
//                         subtitle: Text("Doc ID: ${item.docId} | Score: ${item.score}"),
//                       ),
//                     );
//                   },
//                 );
//               }),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../controller/SearchController.dart';

class SearchScreen extends StatelessWidget {
  final tfidfController controller = Get.put(tfidfController());
  final TextEditingController textController = TextEditingController();

  void _handleSearch() {
    final group = controller.selectedGroup.value;
    print('Selected group: $group');
    final method = controller.selectedMethod.value;
    final query = controller.queryText.value;
    final isIndexed = controller.useInvertedIndex.value;

    controller.search(query: query, method: method, group: group, isIndexed: isIndexed);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("🔍 IR Search Engine")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: Obx(() => DropdownButton<String>(
                    value: controller.selectedGroup.value,
                    onChanged: (val) => controller.selectedGroup.value = val!,
                    items: ['quora', 'webis']
                        .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                        .toList(),
                  )),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Obx(() => DropdownButton<String>(
                    value: controller.selectedMethod.value,
                    onChanged: (val) => controller.selectedMethod.value = val!,
                    items: ['tfidf', 'bert', 'hybrid', 'bm25', 'vector store','bm25hybrid','hybrid_vector']
                        .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                        .toList(),
                  )),
                ),
              ],
            ),
            Obx(() => CheckboxListTile(
              title: const Text("استخدام الفهرسة العكسية"),
              value: controller.useInvertedIndex.value,
              onChanged: (val) => controller.useInvertedIndex.value = val!,
            )),
            const SizedBox(height: 16),
            TextField(
              controller: textController,
              decoration: const InputDecoration(
                labelText: "أدخل الاستعلام",
                border: OutlineInputBorder(),
              ),
              onChanged: (val) => controller.queryText.value = val,
              onSubmitted: (_) => _handleSearch(),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _handleSearch,
              icon: const Icon(Icons.search),
              label: const Text("ابحث الآن"),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: Obx(() {
                if (controller.isLoading.value) {
                  return const Center(child: CircularProgressIndicator());
                }

                // إذا كانت الطريقة "vector store" نعرض من قائمة مختلفة
                if (controller.selectedMethod.value == 'vector store') {
                  if (controller.vector_results.isEmpty) {
                    return const Center(child: Text("لا توجد نتائج"));
                  }
                  return ListView.builder(
                    itemCount: controller.vector_results.length,
                    itemBuilder: (context, index) {
                      final item = controller.vector_results[index];
                      return Card(
                        elevation: 3,
                        margin: const EdgeInsets.symmetric(vertical: 6),
                        child: ListTile(
                          leading: CircleAvatar(child: Text("${item.rank}")),
                          title: Text(item.content),
                          subtitle: Text("Doc ID: ${item.docId} | Distance: ${item.score.toStringAsFixed(4)}"),
                        ),
                      );
                    },
                  );
                }

                // لباقي الطرق نستخدم قائمة النتائج العادية
                if (controller.results.isEmpty) {
                  return const Center(child: Text("لا توجد نتائج"));
                }

                return ListView.builder(
                  itemCount: controller.results.length,
                  itemBuilder: (context, index) {
                    final item = controller.results[index];
                    return Card(
                      elevation: 3,
                      margin: const EdgeInsets.symmetric(vertical: 6),
                      child: ListTile(
                        leading: CircleAvatar(child: Text("${item.rank}")),
                        title: Text(item.content),
                        subtitle: Text("Doc ID: ${item.docId} | Score: ${item.score.toStringAsFixed(4)}"),
                      ),
                    );
                  },
                );
              }),
            ),
          ],
        ),
      ),
    );
  }
}