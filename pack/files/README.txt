PupilAD Pack
============

A self-contained copy of the PupilAD app for Windows (64-bit). It includes its own
Python, so the computer needs no Python and no internet.

Start
-----
- On this computer: double-click "Start PupilAD.bat". The app opens in the browser at
  http://127.0.0.1:5050/
- For phones and tablets: double-click "Start PupilAD for phones and tablets.bat" and open
  the link it prints on a device on the same Wi-Fi. If Windows asks about network access,
  choose "Private networks" and Allow.
- Keep the black window open while using the app. Close it to stop.

Your data
---------
Sessions are saved in the "data" folder next to this file. Each session with video is about
300-400 MB, so use a drive with plenty of free space. Back up the "data" folder; it is never
replaced by an update.

Updating
--------
VERSION.txt shows which code version this pack contains. To update, either:
- On the computer with the PupilAD repository, run pack\build_pack.bat. It refreshes this
  folder with the latest code (only the code changes, so it takes a few seconds), or
- Download PupilAD-Pack.zip from the "PupilAD Pack (latest)" release on GitHub and replace
  everything except the "data" folder.

Copy the whole folder, not just the .bat files.
