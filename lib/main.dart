import 'dart:convert';
import 'dart:typed_data';
import 'dart:html' as html;
import 'package:excel/excel.dart' as ex;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:fl_chart/fl_chart.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark().copyWith(
        cardColor: const Color(0xFF1E1E1E),
        scaffoldBackgroundColor: const Color(0xFF121212),
        colorScheme: ColorScheme.dark(
          primary: Colors.blue[400]!,
          secondary: Colors.tealAccent,
          surface: const Color(0xFF1E1E1E),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue[700],
            foregroundColor: Colors.white,
          ),
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Uint8List? fileBytes;
  String fileName = '';
  String message = '';
  bool loading = false;
  Map<String, dynamic>? results;
  List<String> columns = [];
  bool showPiecewise = true;

  Future<void> pickFile() async {
    final input = html.FileUploadInputElement();
    input.accept = '.xlsx,.xls';
    input.click();
    input.onChange.listen((event) async {
      final files = input.files;
      if (files != null && files.isNotEmpty) {
        final file = files[0];
        final reader = html.FileReader();
        reader.readAsArrayBuffer(file);
        await reader.onLoadEnd.first;

        setState(() {
          fileBytes = reader.result as Uint8List;
          fileName = file.name;
          message = 'Selected: $fileName';
        });

        // Parse Excel to get column names using alias 'ex'
        final excel = ex.Excel.decodeBytes(fileBytes!);
        final sheet = excel.tables[excel.tables.keys.first]!;
        if (sheet.rows.isNotEmpty) {
          final firstRow = sheet.rows[0];
          columns = firstRow.map((cell) => cell?.value.toString() ?? '').toList();
          if (columns.every((c) => c.isEmpty)) {
            columns = List.generate(firstRow.length, (i) => 'Column ${i + 1}');
          }
        } else {
          columns = [];
        }
      }
    });
  }

  Future<Map<String, int>?> showColumnSelectionDialog() async {
    int? xColumn;
    int? yColumn;

    return showDialog<Map<String, int>>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Select Columns'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              DropdownButtonFormField<int>(
                decoration: const InputDecoration(labelText: 'X Column'),
                items: List.generate(
                  columns.length,
                  (index) => DropdownMenuItem(
                    value: index,
                    child: Text(columns[index]),
                  ),
                ),
                onChanged: (value) => xColumn = value,
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<int>(
                decoration: const InputDecoration(labelText: 'Y Column'),
                items: List.generate(
                  columns.length,
                  (index) => DropdownMenuItem(
                    value: index,
                    child: Text(columns[index]),
                  ),
                ),
                onChanged: (value) => yColumn = value,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () {
                if (xColumn != null && yColumn != null) {
                  Navigator.pop(context, {
                    'x': xColumn!,
                    'y': yColumn!,
                  });
                }
              },
              child: const Text('OK'),
            ),
          ],
        );
      },
    );
  }

  Future<void> uploadFile() async {
    if (fileBytes == null) {
      setState(() => message = 'No file selected!');
      return;
    }

    setState(() => loading = true);

    try {
      final selection = await showColumnSelectionDialog();

      if (selection == null) {
        setState(() {
          loading = false;
          message = 'Column selection cancelled';
        });
        return;
      }

      var uri = Uri.parse('http://127.0.0.1:8000/process');
      var request = http.MultipartRequest('POST', uri);

      var formData = http.MultipartFile.fromBytes(
        'file',
        fileBytes!,
        filename: fileName,
      );
      request.files.add(formData);

      request.fields['x_column'] = selection['x'].toString();
      request.fields['y_column'] = selection['y'].toString();

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        setState(() {
          results = jsonDecode(response.body);
          message = 'Success!';
        });
      } else {
        setState(() => message = 'Upload failed: ${response.statusCode}');
      }
    } catch (e) {
      setState(() => message = 'Error: $e');
    } finally {
      setState(() => loading = false);
    }
  }

  List<FlSpot> toSpots(List<dynamic> xs, List<dynamic> ys) {
    return List.generate(xs.length, (i) {
      final dx = xs[i];
      final dy = ys[i];
      final x = (dx is num) ? dx.toDouble() : double.parse(dx.toString());
      final y = (dy is num) ? dy.toDouble() : double.parse(dy.toString());
      return FlSpot(x, y);
    });
  }

  Widget buildChart(String title, List<dynamic> xs, List<dynamic> ys) {
    return SizedBox(
      height: 300,
      child: Card(
        margin: const EdgeInsets.all(8),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Text(title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white70,
                  )),
              const SizedBox(height: 8),
              Expanded(
                child: LineChart(
                  LineChartData(
                    lineBarsData: [
                      LineChartBarData(
                        spots: toSpots(xs, ys),
                        isCurved: true,
                        color: Colors.lightBlue[300],
                        dotData: FlDotData(show: false),
                        barWidth: 3,
                      )
                    ],
                    titlesData: FlTitlesData(
                      leftTitles: AxisTitles(
                          sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        getTitlesWidget: (value, meta) {
                          return Text(
                            value.toStringAsFixed(1),
                            style: const TextStyle(color: Colors.white70),
                          );
                        },
                      )),
                      bottomTitles: AxisTitles(
                          sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 30,
                        getTitlesWidget: (value, meta) {
                          return Text(
                            value.toStringAsFixed(1),
                            style: const TextStyle(color: Colors.white70),
                          );
                        },
                      )),
                      topTitles:
                          AxisTitles(sideTitles: SideTitles(showTitles: false)),
                      rightTitles:
                          AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    ),
                    lineTouchData: LineTouchData(enabled: false),
                    gridData: FlGridData(
                      show: true,
                      getDrawingHorizontalLine: (value) => FlLine(
                        color: Colors.white10,
                        strokeWidth: 1,
                      ),
                      getDrawingVerticalLine: (value) => FlLine(
                        color: Colors.white10,
                        strokeWidth: 1,
                      ),
                    ),
                    borderData: FlBorderData(
                      show: true,
                      border: Border.all(color: Colors.white24),
                    ),
                    backgroundColor: const Color(0xFF1E1E1E),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget buildEquation(String title, List<dynamic>? eq) {
    if (eq == null || eq.isEmpty) return const SizedBox();

    String equationText = '';
    if (eq.length > 1) {
      equationText = eq[showPiecewise ? 0 : 1];
    } else {
      equationText = eq[0];
    }

    return Card(
      margin: const EdgeInsets.all(8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('$title Equation:',
                    style: const TextStyle(
                        fontSize: 16, fontWeight: FontWeight.bold)),
                if (eq.length > 1)
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        showPiecewise = !showPiecewise;
                      });
                    },
                    child:
                        Text(showPiecewise ? 'Show Best Fit' : 'Show Piecewise'),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            SelectableText(
              equationText,
              style: const TextStyle(
                fontSize: 14,
                fontFamily: 'Monospace',
                color: Colors.white70,
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Excel Grapher')),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 10),
            ElevatedButton(onPressed: pickFile, child: const Text('Select Excel File')),
            const SizedBox(height: 10),
            ElevatedButton(
                onPressed: loading ? null : uploadFile,
                child: loading ? const CircularProgressIndicator() : const Text('Process File')),
            const SizedBox(height: 10),
            Text(message),
            if (results != null) ...[
              buildChart('Spline', results!['spline_x'], results!['spline_y']),
              buildEquation('Spline', results!['spline_model']),
              buildChart('Derivative', results!['der_x'], results!['der_y']),
              buildEquation('Derivative', results!['der_model']),
              buildChart('Integral', results!['int_x'], results!['int_y']),
              buildEquation('Integral', results!['int_model']),
            ]
          ],
        ),
      ),
    );
  }
}