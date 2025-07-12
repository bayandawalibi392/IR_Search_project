import 'package:get/get.dart';
import 'package:ir_project/view/ir_screen.dart';

import 'constant/reoutes.dart';

List<GetPage<dynamic>>? routes = [
  GetPage(name: AppRout.ir_screen ,page: () =>  SearchScreen()),
];

