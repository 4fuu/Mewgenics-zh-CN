package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	mewpatch "github.com/4fuu/Mewgenics-zh-CN"
)

const textDir = "extracted/data/text"

var csvFiles = []string{
	"misc.csv",
	"additions3.csv",
	"additions2.csv",
	"additions.csv",
	"pronouns.csv",
	"weather.csv",
	"teamnames.csv",
	"progression.csv",
	"keyword_tooltips.csv",
	"cutscene_text.csv",
	"furniture.csv",
	"mutations.csv",
	"enemy_abilities.csv",
	"units.csv",
	"passives.csv",
	"items.csv",
	"events.csv",
	"abilities.csv",
	"npc_dialog.csv",
}

var zhLanguageMeta = map[string]string{
	"CURRENT_LANGUAGE_NAME":     "中文",
	"CURRENT_LANGUAGE_SHIPPABLE": "yes",
}

func cmdApplyTranslations() {
	if _, err := os.Stat(textDir); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "错误: 找不到 %s，请先执行 extract\n", textDir)
		os.Exit(1)
	}

	// 加载内嵌翻译数据
	var progress map[string]string
	if err := json.Unmarshal(mewpatch.TranslationProgressJSON, &progress); err != nil {
		fmt.Fprintf(os.Stderr, "错误: 解析翻译数据失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("已加载 %d 条翻译\n", len(progress))

	totalApplied := 0
	for _, csvFile := range csvFiles {
		fp := filepath.Join(textDir, csvFile)
		if _, err := os.Stat(fp); os.IsNotExist(err) {
			continue
		}

		header, rows, err := readCSV(fp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  读取 %s 失败: %v\n", csvFile, err)
			continue
		}

		// 添加 zh 列
		zhIdx := indexOf(header, "zh")
		if zhIdx < 0 {
			header = append(header, "zh")
			zhIdx = len(header) - 1
			for i := range rows {
				rows[i] = append(rows[i], "")
			}
			fmt.Printf("  %s: 已添加 zh 列\n", csvFile)
		}

		// 应用翻译
		enIdx := indexOf(header, "en")
		if enIdx < 0 {
			continue
		}

		applied := 0
		for i, row := range rows {
			for len(row) < len(header) {
				row = append(row, "")
				rows[i] = row
			}
			key := row[0]
			if strings.HasPrefix(key, "//") || strings.TrimSpace(key) == "" {
				continue
			}
			fullKey := csvFile + "::" + key
			if zh, ok := progress[fullKey]; ok && zh != "" {
				rows[i][zhIdx] = zh
				applied++
			}
		}

		// additions.csv 特殊处理：设置语言元数据
		if csvFile == "additions.csv" {
			for i, row := range rows {
				for len(row) < len(header) {
					row = append(row, "")
					rows[i] = row
				}
				key := row[0]
				if val, ok := zhLanguageMeta[key]; ok && strings.TrimSpace(rows[i][zhIdx]) == "" {
					rows[i][zhIdx] = val
					applied++
				}
			}
		}

		if err := writeCSV(fp, header, rows); err != nil {
			fmt.Fprintf(os.Stderr, "  写入 %s 失败: %v\n", csvFile, err)
			continue
		}

		totalApplied += applied
		fmt.Printf("  %s: 应用了 %d 条翻译\n", csvFile, applied)
	}

	fmt.Printf("完成。共应用 %d 条翻译\n", totalApplied)
}

func readCSV(path string) (header []string, rows [][]string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	// 跳过 UTF-8 BOM
	bom := make([]byte, 3)
	n, _ := f.Read(bom)
	if n < 3 || bom[0] != 0xEF || bom[1] != 0xBB || bom[2] != 0xBF {
		f.Seek(0, io.SeekStart)
	}

	r := csv.NewReader(f)
	r.LazyQuotes = true
	r.FieldsPerRecord = -1

	header, err = r.Read()
	if err != nil {
		return nil, nil, err
	}

	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		rows = append(rows, record)
	}
	return header, rows, nil
}

func writeCSV(path string, header []string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// 写 UTF-8 BOM
	f.Write([]byte{0xEF, 0xBB, 0xBF})

	w := csv.NewWriter(f)
	w.UseCRLF = false
	if err := w.Write(header); err != nil {
		return err
	}
	for _, row := range rows {
		if err := w.Write(row); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func indexOf(slice []string, s string) int {
	for i, v := range slice {
		if strings.EqualFold(v, s) {
			return i
		}
	}
	return -1
}
