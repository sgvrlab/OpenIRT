using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Microsoft.Win32;
using System.IO;
using System.Diagnostics;

namespace EZBuilder
{
    public partial class EZBuilder : Form
    {
        private RegistryKey key;
        private string[] argBuilder, argPost;

        public EZBuilder()
        {
            InitializeComponent();
            key = Microsoft.Win32.Registry.CurrentUser.OpenSubKey("EZBuilder", true);
            if(key == null)
                key = Microsoft.Win32.Registry.CurrentUser.CreateSubKey("EZBuilder");

            try
            {
                textBoxBuilder.Text = (string)key.GetValue("BVH builder");
                textBoxPost.Text = (string)key.GetValue("Post processor");
                textBoxWorking.Text = (string)key.GetValue("Working directory");

                if (textBoxWorking.Text == "")
                    MessageBox.Show("Working directory shoule be set first.");

                Directory.SetCurrentDirectory(textBoxWorking.Text);
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message);
            }

            argBuilder = new string[3] { "", "test.ply", "" };
            argPost = new string[5] { "test.ply", "31", "6", "8", "5" };

            checkAll();
        }

        private void updateRegistry()
        {
            try
            {
                key.SetValue("BVH builder", textBoxBuilder.Text);
                key.SetValue("Post processor", textBoxPost.Text);
                key.SetValue("Working directory", textBoxWorking.Text);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void buttonBuilder_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "Executive Files(*.exe)|*.exe|All files(*.*)|*.*";
            dlg.RestoreDirectory = true;

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                textBoxBuilder.Text = dlg.FileName;
                updateRegistry();
            }
        }

        private void buttonPost_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "Executive Files(*.exe)|*.exe|All files(*.*)|*.*";
            dlg.RestoreDirectory = true;

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                textBoxPost.Text = dlg.FileName;
                updateRegistry();
            }
        }

        private void buttonWorking_Click(object sender, EventArgs e)
        {
            if (folderBrowserDialogWorking.ShowDialog() == DialogResult.OK)
            {
                textBoxWorking.Text = folderBrowserDialogWorking.SelectedPath;
                Directory.SetCurrentDirectory(textBoxWorking.Text);
                updateRegistry();
            }
        }

        private void Run_Click(object sender, EventArgs e)
        {
            updateRegistry();

            bool hasList = false;

            if (textBoxTrans.Text.Length > 0)
                hasList = listBoxSources.Items.Count > 0;
            else
                hasList = listBoxSources.Items.Count > 1;

            try
            {
                if (hasList)
                {
                    if (textBoxName.Text.Length == 0)
                        throw new System.IO.InvalidDataException("You need to specify name");

                    StreamWriter writer = new StreamWriter(textBoxName.Text);
                    foreach (string file in listBoxSources.Items)
                    {
                        writer.WriteLine(file + " " + textBoxTrans.Text);
                    }
                    writer.Flush();
                    writer.Close();
                }

                //Process listProcess = Process.Start("notepad.exe", textBoxName.Text);

                Process buildProcess = Process.Start(textBoxBuilder.Text, textBoxArgBuilder.Text);
                buildProcess.WaitForExit();
                buildProcess.Close();

                Process postProcess = Process.Start(textBoxPost.Text, textBoxArgPost.Text);
                postProcess.WaitForExit();
                postProcess.Close();

                if (hasList)
                {
                    System.IO.File.Delete(textBoxName.Text);
                }

                MessageBox.Show("Done");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private bool isOBJFile(string file)
        {
            int start = Math.Max(file.LastIndexOf('\\'), file.LastIndexOf('/'));
            int end = file.LastIndexOf('.');

            return file.Substring(end + 1).Equals("OBJ", StringComparison.OrdinalIgnoreCase);
        }

        private string getShortFile(string file)
        {
            int start = Math.Max(file.LastIndexOf('\\'), file.LastIndexOf('/'));
            int end = file.Length;

            return file.Substring(start + 1, end - start - 1);
        }

        private void updateAttributes()
        {
            bool hasList = false;
            bool hasTrans = false;
            bool isIncore = false;
            bool useFileMat = false;
            bool removeIndex = false;
            bool generateHCCMesh = false;
            bool generateMTL = false;
            bool useGPUFriendly = false;
            bool generateASVO = false;

            if (textBoxTrans.Text.Length > 0)
                hasList = listBoxSources.Items.Count > 0;
            else
                hasList = listBoxSources.Items.Count > 1;

            hasTrans = textBoxTrans.Text.Length > 0;

            isIncore = !checkBoxMassive.Checked;
            useFileMat = checkBoxUseFileMat.Checked;

            removeIndex = checkBoxRemoveIndex.Checked;
            generateHCCMesh = checkBoxHCCMesh.Checked;
            generateMTL = checkBoxMTL.Checked;
            useGPUFriendly = checkBoxGPU.Checked;
            generateASVO = checkBoxASVO.Checked;

            argBuilder[0] = "";
            if (hasList || hasTrans || isIncore || useFileMat)
            {
                argBuilder[0] = "-";
                argBuilder[0] += hasList ? "f" : "";
                argBuilder[0] += hasTrans ? "m" : "";
                argBuilder[0] += isIncore ? "i" : "";
                argBuilder[0] += useFileMat ? "x" : "";
            }

            if (listBoxSources.Items.Count == 1)
                textBoxSourceFile.Text = listBoxSources.Items[0].ToString();
            else
                textBoxSourceFile.Text = "";

            if (listBoxSources.Items.Count > 0)
            {
                argBuilder[1] = hasList ? textBoxName.Text : listBoxSources.Items[0].ToString();
            }
            else
                argBuilder[1] = "test.ply";

            argBuilder[2] = textBoxMTLFile.Text;

            if (isOBJFile(argBuilder[1]))
            {
                argPost[0] = getShortFile(argBuilder[1]);
                checkBoxMTL.Checked = false;
                generateMTL = false;
            }
            else
                argPost[0] = argBuilder[1];

            uint flag = 0;
            flag |= (uint)((removeIndex ? 1 : 0) << 0);
            flag |= (uint)((generateHCCMesh ? 1 : 0) << 1);
            flag |= (uint)((generateMTL ? 1 : 0) << 2);
            flag |= (uint)((generateASVO ? 1 : 0) << 3);
            flag |= (uint)((useGPUFriendly ? 1 : 0) << 4);
            argPost[1] = ""+flag;

            textBoxASVOOption1.ReadOnly = !generateASVO;

            argPost[2] = textBoxASVOOption1.Text;
            argPost[3] = textBoxASVOOption2.Text;
            argPost[4] = textBoxASVOOption3.Text;

            textBoxName.ReadOnly = !hasList;

            textBoxArgBuilder.Text = "";
            for (int i = 0; i < argBuilder.Length; i++)
            {
                textBoxArgBuilder.Text += argBuilder[i];
                if (i + 1 < argBuilder.Length && argBuilder[i + 1].Length > 0)
                    textBoxArgBuilder.Text += " ";
            }

            textBoxArgPost.Text = "";
            for (int i = 0; i < argPost.Length; i++)
            {
                textBoxArgPost.Text += argPost[i];
                if (i + 1 < argPost.Length && argPost[i + 1].Length > 0)
                    textBoxArgPost.Text += " ";
            }
        }

        private void checkAll()
        {
            checkBoxRemoveIndex.Checked = true;
            checkBoxHCCMesh.Checked = true;
            checkBoxMTL.Checked = true;
            checkBoxGPU.Checked = true;
            checkBoxASVO.Checked = true;

            updateAttributes();
        }

        private void addSourceFileToList(string file)
        {
            try
            {
                System.Uri uri1 = new Uri(textBoxWorking.Text + "\\boo");
                System.Uri uri2 = new Uri(file);

                Uri relativeUri = uri1.MakeRelativeUri(uri2);
                file = relativeUri.ToString();

                listBoxSources.Items.Add(file);
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message);
            }
        }

        private void addSourceFile(string file)
        {
            // get the file attributes for file or directory
            try
            {
                FileAttributes attr = File.GetAttributes(file);

                //detect whether its a directory or file
                if ((attr & FileAttributes.Directory) == FileAttributes.Directory)
                {
                    if (textBoxName.Text.Length == 0)
                    {
                        string[] temp = file.Split(new char[2] {'\\', '/'});
                        textBoxName.Text = temp[temp.Length-1];
                    }
                    walkDirectoryTree(new DirectoryInfo(file));
                }
                else
                    addSourceFileToList(file);
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message);
                return;
            }

        }

        private void listBoxSources_DragDrop(object sender, DragEventArgs e)
        {
            string[] files = (string[])e.Data.GetData(DataFormats.FileDrop, false);
            foreach (string file in files)
                addSourceFile(file);
            updateAttributes();
        }

        private void listBoxSources_DragEnter(object sender, DragEventArgs e)
        {
            e.Effect = DragDropEffects.All;
        }

        private void walkDirectoryTree(DirectoryInfo root)
        {
            FileInfo[] files = null;
            DirectoryInfo[] subDirs = null;

            // First, process all the files directly under this folder 
            try
            {
                files = root.GetFiles("*.ply");
            }
            // This is thrown if even one of the files requires permissions greater 
            // than the application provides. 
            catch (UnauthorizedAccessException e)
            {
                // This code just writes out the message and continues to recurse. 
                // You may decide to do something different here. For example, you 
                // can try to elevate your privileges and access the file again.
                MessageBox.Show(e.Message);
            }

            catch (DirectoryNotFoundException e)
            {
                MessageBox.Show(e.Message);
            }

            if (files != null)
            {
                foreach (FileInfo fi in files)
                {
                    // In this example, we only access the existing FileInfo object. If we 
                    // want to open, delete or modify the file, then 
                    // a try-catch block is required here to handle the case 
                    // where the file has been deleted since the call to TraverseTree().
                    addSourceFileToList(fi.FullName);
                }

                // Now find all the subdirectories under this directory.
                subDirs = root.GetDirectories();

                foreach (DirectoryInfo dirInfo in subDirs)
                {
                    // Resursive call for each subdirectory.
                    walkDirectoryTree(dirInfo);
                }
            }
        }

        private void buttonClear_Click(object sender, EventArgs e)
        {
            textBoxName.Text = "";
            listBoxSources.Items.Clear();
            updateAttributes();
        }

        private void buttonAddSourceFile_Click(object sender, EventArgs e)
        {
            textBoxSourceFile.Text = "";
            if(textBoxSourceFile.Text.Length > 0)
                addSourceFile(textBoxSourceFile.Text);
            updateAttributes();
        }

        private void listBoxSources_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Delete)
            {
                for(int i=0;i<listBoxSources.SelectedItems.Count;i++)
                {
                    listBoxSources.Items.Remove(listBoxSources.SelectedItems[i].ToString());
                    i--;
                }
                updateAttributes();
            }
        }

        private void textBoxWorking_TextChanged(object sender, EventArgs e)
        {
        }

        private void textBoxWorking_Leave(object sender, EventArgs e)
        {
            try
            {
                Directory.SetCurrentDirectory(textBoxWorking.Text);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }

        private void textBoxTrans_TextChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void buttonMTLFile_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "MTL Files(*.mtl)|*.mtl|All files(*.*)|*.*";
            dlg.RestoreDirectory = true;

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                textBoxMTLFile.Text = dlg.FileName;

                checkBoxMTL.Checked = false;

                updateAttributes();
            }
        }

        private void textBoxMTLFile_TextChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void buttonIdentity_Click(object sender, EventArgs e)
        {
            textBoxTrans.Text = "";
        }

        private void buttonYZ_Click(object sender, EventArgs e)
        {
            textBoxTrans.Text = "1 0 0 0 0 0 -1 0 0 1 0 0 0 0 0 1";
        }

        private void checkBoxMassive_CheckedChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void textBoxName_TextChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void buttonFindAddFile_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "Model Files(*.ply)|*.ply|All files(*.*)|*.*";
            dlg.RestoreDirectory = true;

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                textBoxSourceFile.Text = dlg.FileName;
            }
        }

        private void checkBoxUseFileMat_CheckedChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void checkBoxRemoveIndex_CheckedChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void checkBoxHCCMesh_CheckedChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void checkBoxMTL_CheckedChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void checkBoxGPU_CheckedChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void checkBoxASVO_CheckedChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void buttonAll_Click(object sender, EventArgs e)
        {
            checkAll();
        }

        private void textBoxASVOOption1_TextChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void textBoxASVOOption2_TextChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void textBoxASVOOption3_TextChanged(object sender, EventArgs e)
        {
            updateAttributes();
        }

        private void textBoxSourceFile_TextChanged(object sender, EventArgs e)
        {
        }
    }
}
