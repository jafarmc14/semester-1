Your task is to prepare 2 shell scripts:

1. Write a script  chname  which will have the following syntax:

  chname  [-r|--recursive] [-s|--subdirectories] [-l|--lowercase|-u|-uppercase] <dir/file names...>
  chname  [-h|--help]
  

The goal of the script is to change names of files. The script is dedicated to
lowerizing (-l or --lowercase) file and directory names or uppercasing (-u or
--uppercase) file and directory names given as arguments. 

Changes may be done either with recursion (for all the files in subdirectories
'-r' or --recursive) or without it.  In recursive mode changes may affect only
regular file names  or  subdirectory names (if with '-s' or --subdirectories)
as well.  Option -s without -r allows modification of directory names in the
current directory. Option -h (or --help) should print help message.


2. Write additional script, chname_examples, to demostrate and to test chname
working scenarios, also to test how the script behaves when incorrect arguments
to chname are given. This script should create some set of directories and
files for demonstration, and be prepared ahead of a person in charge of
laboratory evaluation to demonstrate behaviour of your basic chname script
executed with different - correct and incorrect - options and file names, also
for non-trivial scenarious.


--
Examples of correct usage:

chname -l File1 FILE2
chname -u a*
chname -r -u file1 dir1 dir2 dir3
chname -r -s -u file1 file2 dir1 dir2
