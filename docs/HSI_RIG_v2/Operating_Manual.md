# HSI Rig Operation Manual (W.I.P.)

## 1. What this document covers
  This document describes how to safely operate the Hyperspectral Scanning (HSI) Rig to acquire spectral image data. It will cover the pre-scan checks, system startup, software operation, and post-scan procedures.

  The rig consists of a large 4040 frame with:

  * A moving bed on the Y-Axis for moving the subject under the camera for imaging
  * An adjustable Z-axis that the camera assembly and lighting are mounted to for focusing and working distance control
  * A hyperspectral camera (FX10e) is mounted above the scanning area on the gantry that is moved along the Z-axis
  * A control PC running windows 11 will run the dedicated GUI software used for interacting with the rig for controlling the scan motion and image acquisition

---

## 2. Safety and Handling

* Ensure power is disconnected before making hardware adjustments.
* During operation ensure **E-Stop is withing reach at all time** in the event of a problem
  * The **E-Stop cuts all power to the rig** (except the camera and pc)
* Keep hands clear of the moving Y-bed and Z-Axis during operation.
* Do not open the control box when powered on and only if you know what you are doing.
* Do not touch the light when on the system is on or after operating has halted, as they could cause injury via a burn.
* Diffusers attached to the lighting should be handled with care, as there are glass and should they break could lead to cuts.

---

## 3. Pre-Scan Hardware Checks

Perform these checks before every scanning session.

### 3.1 Power and Connectivity

* Verify all power cables and communication cables are connected, undamaged and will not get pinched/traps in the rig during operations.
* Check that the system control PC detects all connected devices, ie camera and rig controller.

### 3.2 Optical System

* Ensure lense caps are off and stores safely before scanning
  * **Put all caps back on once finished using the machine**
* Inspect lenses, filters for dust or fingerprints.
* Check to make sure all 4 diffuser glass slides are secure in their mounts on the lights.
* The **calibration strips are expensive**, therefore when handling calibrations strips:
  * **Ensure you are where latex (or equivilent) gloves**
  * **Handling the strip via the edges**, to prevent surface damages **avoiding touching the top surface of strip**.
  * Confirm that calibration strips are clean and in place correctly.
  * **DO NOT PLACE SCANNING SUBJECT ON TOP OF THE STRIPS**

### 3.3 Mechanical Components

* Check Y-Axis belts:
  * are tight and do not have excess slack.
  * are not damaged.
  * Ensure the limit switch is:
    * not damaged.
    * mounted securely to side of the frame
      * ***In the event that the switch is not positioned correctly, refere to the [maintance guide](./Maintance.md) for detail on fixing***
* Check Z-Axis:
  * Make sure the lead screws either side are not excessively bent out of shape.
  * Lead screws for any debris in the threads.
  * Gantery mount block are not damages, ie. bent or cracked.
  * Gantry is level before scanning
    * If not level the user may rotate the left and/or right motors **when the rig is unpowered** to level out the gantry
    * ***An uneven gantry could result in a skewed image scan***
  * Ensure the limit switch is:
    * not damaged.
    * mounted securely to side of the frame
      * ***In the event that the switch is not positioned correctly, refere to the [maintance guide](./Maintance.md) for detail on fixing***
  * Check that the bottom stops are attached securely so that in the event that the gantery is overdrive or falls torwards the bed that it is stopped before the camera is crush  
* Make sure there are no obstructions along the axis travel paths.

### 3.4 Environmental Conditions

* Confirm ambient light conditions are suitable (if relevant).
  * Reduction of ambient light is almost always desired and therefore is best in a dark room
* Avoid depositing and leaving dust and particulates as much as possible on to the scanning surface
  * **DO NOT PLACE SCANNING SUBJECT ON TOP OF THE STRIPS**
  * Clean the bed after done scanning
    * The bed plates are removeable and therefore make wiping down and disinfecting (if required) easier
* Do not eat or drink when operating the scanning rig

---

## 4. Software Setup and Scan Operation

### 4.1 Starting the System

* Power the scanning rig in order:
  1. Rig control box
     * red power switch on left side of the box
     * Ensure the E-Stop is not engaged
  2. Rig lights
     * black power switch on farthers right side of the box
  3. Camera
     * Plugging in the camera
  4. Control PC

### 4.2 Operating the GUI for a scan

1. start up the HSI GUI
2. Under the **camera** section:
   1. Connect to the FX10e camera
   2. Specify our output directory for image output
3. Under the **adjust** section:
   1. Set your shutter settings
   2. Request your required frame rate
      * *This is used to calculate the required movement speed for the rig's scannign bed*
   3. Set the camera exposure
   4. Specify the **Spectral** and **Spatial** binning
4. Under the **Rig Controller Connection** section:
   1. Select from the dropdown the COM port for the scanning rig controller
      * *This is typically COM3 for the PC attached to the rig, but this could differ*
5. Under the **Acquisition** section:
   1. Set your Capture Mode
   2. Set your Frame Count
   3. Click the **Start** button to begin scanning
      * Your can stop the scan routine midway through if required by clicking the **Stop** button next to **Start**
   4. Repeat the scan process process 


### 4.3 Post-Scan Procedures


---

## 5. Troubleshooting (Basic)


---

## 6. Appendix

* **Revision History**

---
