https://colab.research.google.com/drive/1ibPONCmdKUyjRJVcOE8fUN61mSMEbM2n?usp=sharing

## pre-requirements:
Enable Long Paths via Group Policy (for Windows Pro and Enterprise users):

    Press Win + R, type gpedit.msc, and press Enter to open the Group Policy Editor.
    Navigate to Computer Configuration > Administrative Templates > System > Filesystem.
    Double-click on the "Enable Win32 long paths" option.
    Set it to "Enabled" and click "OK".

Enable Long Paths via Registry Editor (for Windows Home users):

    Press Win + R, type regedit, and press Enter to open the Registry Editor.
    Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem.
    Find the LongPathsEnabled entry. If it doesn’t exist, create a new DWORD (32-bit) Value named LongPathsEnabled.
    Set the value to 1.
    Restart your computer to apply the changes.
