import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:async'; 
import 'package:intl/intl.dart'; 
import 'package:flutter/services.dart';
import 'package:audioplayers/audioplayers.dart'; // New Import

late List<CameraDescription> _cameras;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await [Permission.camera, Permission.microphone].request();
  _cameras = await availableCameras();
  
  runApp(MaterialApp(
    theme: ThemeData(
      textTheme: GoogleFonts.robotoMonoTextTheme(),
    ),
    home: const DrishtiScreen(),
    debugShowCheckedModeBanner: false,
  ));
}

class DrishtiScreen extends StatefulWidget {
  const DrishtiScreen({super.key});

  @override
  State<DrishtiScreen> createState() => _DrishtiScreenState();
}

class _DrishtiScreenState extends State<DrishtiScreen> {
  late CameraController controller;
  late FlutterVision vision;
  late Timer _timer;
  final AudioPlayer _audioPlayer = AudioPlayer(); // Initialize Player
  
  String _currentTime = "";
  String _currentDate = "";
  bool isLoaded = false;
  bool isDetecting = false;
  bool _isAlarmPlaying = false;
  List<Map<String, dynamic>> yoloResults = [];

  @override
  void initState() {
    super.initState();
    vision = FlutterVision();
    setupCamera();
    
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (mounted) {
        setState(() {
          _currentTime = DateFormat('HH:mm:ss').format(DateTime.now());
          _currentDate = DateFormat('dd/MM/yyyy').format(DateTime.now());
        });
      }
    });
  }

  Future<void> setupCamera() async {
    controller = CameraController(_cameras[0], ResolutionPreset.high, enableAudio: false);
    await controller.initialize();
    await loadModel();
    setState(() {
      isLoaded = true;
    });
  }

  Future<void> loadModel() async {
    await vision.loadYoloModel(
      modelPath: 'assets/model.tflite',
      labels: 'assets/labels.txt',
      modelVersion: 'yolov8',
      numThreads: 6, 
      useGpu: true, 
    );
  }

  @override
  void dispose() {
    _timer.cancel();
    _audioPlayer.dispose(); // Clean up audio
    controller.dispose();
    vision.closeYoloModel();
    super.dispose();
  }

  // --- ALARM LOGIC ---
  void _manageAlarm(bool shouldPlay) async {
    if (shouldPlay && !_isAlarmPlaying) {
      _isAlarmPlaying = true;
      await _audioPlayer.setReleaseMode(ReleaseMode.loop);
      await _audioPlayer.play(AssetSource('alarm.mp3'));
    } else if (!shouldPlay && _isAlarmPlaying) {
      _isAlarmPlaying = false;
      await _audioPlayer.stop();
    }
  }

  Future<void> processCameraImage(CameraImage image) async {
    if (isDetecting || !mounted) return;

    setState(() { isDetecting = true; });

    final result = await vision.yoloOnFrame(
      bytesList: image.planes.map((plane) => plane.bytes).toList(),
      imageHeight: image.height,
      imageWidth: image.width,
      iouThreshold: 0.4,
      confThreshold: 0.35,
      classThreshold: 0.35,
    );

    // High-confidence weapon check
    bool hasWeapon = result.any((res) => 
      (res['tag'] == 'firearm' || res['tag'] == 'knife') && res['box'][4] > 0.45
    );

    // Trigger Alarm and Haptics
    if (hasWeapon) {
      HapticFeedback.heavyImpact();
      _manageAlarm(true);
    } else {
      _manageAlarm(false);
    }

    setState(() {
      yoloResults = result; 
      isDetecting = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (!isLoaded) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: Color(0xFF00FF99))),
      );
    }

    final size = MediaQuery.of(context).size;
    double scaleX = size.width / controller.value.previewSize!.height;
    double scaleY = size.height / controller.value.previewSize!.width;
    
    bool threatPresent = yoloResults.any((res) => 
      (res['tag'] == 'firearm' || res['tag'] == 'knife') && res['box'][4] > 0.45
    );

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          Positioned.fill(child: CameraPreview(controller)),

          // RED ALERT FLASH
          if (threatPresent)
            Positioned.fill(
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.redAccent.withOpacity(0.4), width: 10),
                ),
              ),
            ),

          // BOXES
          ...yoloResults.map((res) {
            final isWeapon = res['tag'] == 'firearm' || res['tag'] == 'knife';
            return Positioned(
              left: res['box'][0] * scaleX,
              top: res['box'][1] * scaleY,
              width: (res['box'][2] - res['box'][0]) * scaleX,
              height: (res['box'][3] - res['box'][1]) * scaleY,
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(
                    color: isWeapon ? Colors.redAccent : const Color(0xFF00FF99),
                    width: 2,
                  ),
                ),
                child: Align(
                  alignment: Alignment.topLeft,
                  child: Container(
                    color: isWeapon ? Colors.redAccent : const Color(0xFF00FF99),
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    child: Text(
                      "${res['tag'].toUpperCase()} ${(res['box'][4] * 100).toStringAsFixed(0)}%",
                      style: GoogleFonts.robotoMono(color: Colors.black, fontSize: 10, fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
              ),
            );
          }),

          // TOP HUD
          Positioned(
            top: 40,
            left: 15,
            right: 15,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("DRISHTI", style: GoogleFonts.robotoMono(color: const Color(0xFF00FF99), fontSize: 28, fontWeight: FontWeight.bold, letterSpacing: 4)),
                    Text("SECTOR: ALPHA-1 | ALARM: ${_isAlarmPlaying ? 'ACTIVE' : 'READY'}", 
                      style: GoogleFonts.robotoMono(color: _isAlarmPlaying ? Colors.redAccent : const Color(0xFF00FF99).withOpacity(0.7), fontSize: 10)),
                  ],
                ),
                Text("$_currentTime\n$_currentDate", textAlign: TextAlign.right, style: GoogleFonts.robotoMono(color: const Color(0xFF00FF99), fontSize: 12)),
              ],
            ),
          ),

          // STATUS BOX
          Center(
            child: Container(
              width: size.width * 0.9,
              height: 80,
              decoration: BoxDecoration(
                border: Border.all(color: !threatPresent ? const Color(0xFF00FF99) : Colors.redAccent, width: 2),
                color: Colors.black.withOpacity(0.3),
              ),
              child: Center(
                child: Text(
                  !threatPresent ? "○ NO THREAT DETECTED" : "● TARGET IDENTIFIED",
                  style: GoogleFonts.robotoMono(color: !threatPresent ? const Color(0xFF00FF99) : Colors.redAccent, fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ),

          // BUTTON
          Positioned(
            bottom: 40,
            left: 20,
            right: 20,
            child: InkWell(
              onTap: () {
                if (!controller.value.isStreamingImages) {
                  controller.startImageStream(processCameraImage);
                } else {
                  controller.stopImageStream();
                  _manageAlarm(false); // Kill alarm if feed stops
                }
                setState(() {});
              },
              child: Container(
                height: 60,
                decoration: BoxDecoration(border: Border.all(color: const Color(0xFF00FF99)), color: Colors.black.withOpacity(0.6)),
                child: Center(
                  child: Text(
                    controller.value.isStreamingImages ? "TERMINATE FEED" : "ACTIVATE FEED",
                    style: GoogleFonts.robotoMono(color: const Color(0xFF00FF99), fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}