on run argv
  set inPath to POSIX file (item 1 of argv)
  set outPath to POSIX file (item 2 of argv)
  with timeout of 600 seconds
  tell application "Keynote"
    set theDoc to open inPath
    delay 2
    export theDoc to outPath as PDF with properties {PDF image quality:Best}
    close theDoc saving no
  end tell
  end timeout
end run
