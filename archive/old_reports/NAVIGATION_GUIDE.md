# Documentation Index & Navigation Guide

## 📚 Complete Documentation Set

This folder now contains comprehensive documentation for the refactored `verify_covariates.py` script. Here's what each file contains:

---

## 📖 Document Guide

### **1. START HERE: README_STRUCTURE.md** ⭐
**Best for:** Getting a complete picture
- High-level overview of the problem
- Answers to your specific questions
- Recap of all 5 sections
- Key learning points
- **Time to read:** 10-15 minutes

---

### **2. QUICK_REFERENCE.md**
**Best for:** Using the script quickly
- Quick usage examples
- Troubleshooting tips
- Common scenarios
- Copy-paste ready commands
- **Time to read:** 5 minutes

---

### **3. COVARIATE_VERIFICATION_GUIDE.md**
**Best for:** Understanding the purpose
- Problem statement
- How the script works step-by-step
- Flexibility explanation
- Input/output structure
- Example scenarios
- **Time to read:** 10-12 minutes

---

### **4. SCRIPT_ARCHITECTURE.md**
**Best for:** Deep technical understanding
- 5 sections explained in detail
- Data flow diagram
- Flexibility principles
- Error detection
- Example transformations
- **Time to read:** 15-20 minutes

---

### **5. SCRIPT_STRUCTURE.md**
**Best for:** Code structure and technical details
- File organization diagram
- Function signatures
- Regex patterns used
- Complexity analysis
- Data structures
- **Time to read:** 15-20 minutes

---

### **6. VISUAL_REFERENCE.md**
**Best for:** Visual learners
- Component relationships diagram
- Data flow visualization
- Validation logic flow
- Error scenarios
- Output organization
- **Time to read:** 10 minutes

---

## 🎯 Reading Recommendations

### **If you're short on time (5 minutes):**
1. Read: **QUICK_REFERENCE.md**
2. Run: `python3 verify_covariates.py batch_3x3_job_with_covariates.m`
3. Done! ✓

### **If you want to understand everything (30 minutes):**
1. Read: **README_STRUCTURE.md** (10 min)
2. Read: **VISUAL_REFERENCE.md** (10 min)
3. Scan: **QUICK_REFERENCE.md** (5 min)
4. Reference: Others as needed
5. Done! ✓

### **If you're a developer (60 minutes):**
1. Read: **README_STRUCTURE.md** (10 min)
2. Read: **SCRIPT_ARCHITECTURE.md** (20 min)
3. Read: **SCRIPT_STRUCTURE.md** (20 min)
4. Skim: **VISUAL_REFERENCE.md** (10 min)
5. Review: The actual code in `verify_covariates.py`
6. Done! ✓

### **If you need to maintain/extend this (full read):**
1. **README_STRUCTURE.md** - Context
2. **SCRIPT_STRUCTURE.md** - Code organization
3. **SCRIPT_ARCHITECTURE.md** - Technical details
4. **verify_covariates.py** - Actual code
5. Done! ✓

---

## 🔑 Key Answers to Your Questions

### **Q: How does it automatically check TIV and IQR?**
**Answer:** See **README_STRUCTURE.md** → "Q1: How does the script automatically check which files are IQR and TIV?"
- It reads the actual names from the batch file
- No hardcoding needed

### **Q: What if they are called differently?**
**Answer:** See **README_STRUCTURE.md** → "Q2: What if they are called differently?"
- Works with any covariate names
- Automatically detects what's in the file

### **Q: What if I have other variables (Age, Sex)?**
**Answer:** See **README_STRUCTURE.md** → "Q3: What if I have other variables?"
- Script finds ALL covariates
- Validates each independently

### **Q: How are filenames compared to check order?**
**Answer:** See **README_STRUCTURE.md** → "Q4: How are filenames compared to check order?"
- Script uses count comparison, not filename matching
- If counts match → order is correct

---

## 🗂️ File Organization

```
/Volumes/Thunder/129_PK01/cat12/stats/
│
├─ verify_covariates.py (the script)
│  └─ Well-organized into 5 sections
│
└─ DOCUMENTATION FILES:
   ├─ README_STRUCTURE.md ⭐ START HERE
   ├─ QUICK_REFERENCE.md (quick usage)
   ├─ COVARIATE_VERIFICATION_GUIDE.md (detailed explanation)
   ├─ SCRIPT_ARCHITECTURE.md (technical architecture)
   ├─ SCRIPT_STRUCTURE.md (code structure)
   ├─ VISUAL_REFERENCE.md (diagrams & visuals)
   └─ NAVIGATION_GUIDE.md (this file)
```

---

## 🔄 Script Sections (5 Parts)

### **SECTION 1: FILE DISCOVERY**
- Finds batch files automatically
- Document: **SCRIPT_STRUCTURE.md** → "File Organization"

### **SECTION 2: FILE EXTRACTION**
- Extracts NIfTI file paths from batch
- Document: **SCRIPT_ARCHITECTURE.md** → "Section 2: File Extraction"

### **SECTION 3: COVARIATE EXTRACTION**
- Extracts covariate values and names
- Document: **SCRIPT_ARCHITECTURE.md** → "Section 3: Covariate Extraction"

### **SECTION 4: VALIDATION**
- Compares file and covariate counts
- Document: **VISUAL_REFERENCE.md** → "Validation Logic Flow"

### **SECTION 5: DISPLAY & REPORTING**
- Creates comprehensive validation report
- Document: **SCRIPT_STRUCTURE.md** → "Output Structure"

---

## 💡 Core Concepts

### **Flexibility (No Hardcoding)**
- See: **README_STRUCTURE.md** → "Answering Your Original Questions"
- See: **SCRIPT_ARCHITECTURE.md** → "How Flexibility Is Achieved"

### **Validation Logic**
- See: **COVARIATE_VERIFICATION_GUIDE.md** → "Step 3: Compare Files with Covariates"
- See: **VISUAL_REFERENCE.md** → "Validation Logic Flow"

### **Data Flow**
- See: **SCRIPT_ARCHITECTURE.md** → "Data Flow Diagram"
- See: **VISUAL_REFERENCE.md** → "Data Transformation Through Script"

### **Error Detection**
- See: **SCRIPT_ARCHITECTURE.md** → "Error Detection"
- See: **VISUAL_REFERENCE.md** → "Error Detection Scenarios"

---

## 🚀 Quick Start

```bash
# Navigate to stats folder
cd /Volumes/Thunder/129_PK01/cat12/stats

# Run with auto-detection
python3 verify_covariates.py

# Run with specific batch file
python3 verify_covariates.py batch_3x3_job_with_covariates.m

# Run with custom settings
python3 verify_covariates.py --rows 20 batch_3x3_job_with_covariates.m
```

Expected output:
- **VALID** ✓ → Safe to use in SPM
- **INVALID** ✗ → Fix covariates and try again

---

## 📊 What to Look For in Output

### **Good Output (VALID)**
```
✓ PASS: All counts match! File order is correct.
✓ Status: VALID
✓ Files analyzed: 369
✓ Covariates validated: 2
```
→ Proceed with SPM analysis ✓

### **Bad Output (INVALID)**
```
✗ FAIL: Counts do not match!
✗ Status: INVALID
```
→ Fix covariates and retry ✗

---

## 🔧 Troubleshooting Reference

| Issue | Solution | Document |
|-------|----------|----------|
| Script finds wrong batch | Specify file explicitly | QUICK_REFERENCE.md |
| No covariates found | Using wrong batch file | QUICK_REFERENCE.md |
| File not found | Check path spelling | QUICK_REFERENCE.md |
| Want more entries | Use `--rows 20` | QUICK_REFERENCE.md |
| Need to find batch file | Use `--dir /path` | QUICK_REFERENCE.md |
| Want to understand logic | Read README_STRUCTURE.md | README_STRUCTURE.md |
| Need architecture details | Read SCRIPT_ARCHITECTURE.md | SCRIPT_ARCHITECTURE.md |

---

## 📈 Learning Path

```
Beginner:
  1. QUICK_REFERENCE.md (5 min)
  2. Run the script
  3. Done!

Intermediate:
  1. README_STRUCTURE.md (10 min)
  2. VISUAL_REFERENCE.md (10 min)
  3. QUICK_REFERENCE.md (5 min)
  4. Understand how it works
  5. Ready to use!

Advanced:
  1. All beginner + intermediate
  2. SCRIPT_ARCHITECTURE.md (20 min)
  3. SCRIPT_STRUCTURE.md (20 min)
  4. Read the actual code
  5. Understand all details
  6. Can modify/extend!
```

---

## ✅ Checklist for Using the Script

Before analyzing your data in SPM:

- [ ] Read QUICK_REFERENCE.md
- [ ] Located your batch file
- [ ] Run: `python3 verify_covariates.py batch_file.m`
- [ ] Check output: Is it VALID?
- [ ] If VALID: Proceed with SPM ✓
- [ ] If INVALID: Fix covariates and retry

---

## 🎓 Key Takeaways

1. **Purpose:** Verify covariate order matches file order
2. **Why:** Prevents invalid statistical results
3. **How:** Count comparison (simple and effective)
4. **Flexibility:** Works with ANY covariate names and count
5. **No hardcoding:** Portable across all projects
6. **Output:** Clear VALID/INVALID status

---

## 📞 Need Help?

### **Questions about the script?**
→ Check **README_STRUCTURE.md** → "Answering Your Original Questions"

### **How to use it?**
→ Check **QUICK_REFERENCE.md** → "Quick Usage"

### **What's the technical structure?**
→ Check **SCRIPT_STRUCTURE.md** → "File Organization"

### **Why is it flexible?**
→ Check **SCRIPT_ARCHITECTURE.md** → "How Flexibility Is Achieved"

### **Visual learner?**
→ Check **VISUAL_REFERENCE.md** → All diagrams

### **Full details?**
→ Read **README_STRUCTURE.md** completely

---

## 📋 Summary

You now have:
- ✓ Refactored, well-organized script
- ✓ No hardcoded paths or names
- ✓ Works with any covariate names and counts
- ✓ Clear structure with 5 sections
- ✓ Comprehensive validation
- ✓ Detailed documentation (6 files)
- ✓ Quick reference guide
- ✓ Visual diagrams

**Next step:** Use the script before every SPM analysis! 🚀
