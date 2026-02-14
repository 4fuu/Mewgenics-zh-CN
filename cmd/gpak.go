package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	gpakPath    = "resources.gpak"
	gpakBackup  = "resources.gpak.bak"
	gpakPatched = "resources_patched.gpak"
	extractDir  = "extracted"
)

type gpakEntry struct {
	Name   string
	Size   uint32
	Offset int64
}

func parseIndex(path string) (header []byte, entries []gpakEntry, dataStart int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, err
	}
	defer f.Close()

	header = make([]byte, 4)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, nil, 0, fmt.Errorf("读取文件头失败: %w", err)
	}

	type rawEntry struct {
		name string
		size uint32
	}
	var raw []rawEntry

	for {
		pos, _ := f.Seek(0, io.SeekCurrent)
		var nameLen uint16
		if err := binary.Read(f, binary.LittleEndian, &nameLen); err != nil {
			f.Seek(pos, io.SeekStart)
			break
		}
		if nameLen == 0 || nameLen > 500 {
			f.Seek(pos, io.SeekStart)
			break
		}
		nameBuf := make([]byte, nameLen)
		if _, err := io.ReadFull(f, nameBuf); err != nil {
			f.Seek(pos, io.SeekStart)
			break
		}
		name := string(nameBuf)
		// 检查是否全部可打印
		valid := true
		for _, c := range name {
			if c < 0x20 || c == 0x7f {
				valid = false
				break
			}
		}
		if !valid {
			f.Seek(pos, io.SeekStart)
			break
		}
		var size uint32
		if err := binary.Read(f, binary.LittleEndian, &size); err != nil {
			f.Seek(pos, io.SeekStart)
			break
		}
		raw = append(raw, rawEntry{name, size})
	}

	dataStart, _ = f.Seek(0, io.SeekCurrent)
	current := dataStart
	for _, r := range raw {
		entries = append(entries, gpakEntry{Name: r.name, Size: r.size, Offset: current})
		current += int64(r.size)
	}
	return header, entries, dataStart, nil
}

func cmdInfo() {
	header, entries, dataStart, err := parseIndex(gpakPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: %v\n", err)
		os.Exit(1)
	}

	fi, _ := os.Stat(gpakPath)
	headerVal := binary.LittleEndian.Uint32(header)
	fmt.Printf("文件: %s\n", gpakPath)
	fmt.Printf("大小: %d 字节 (%.1f MB)\n", fi.Size(), float64(fi.Size())/1024/1024)
	fmt.Printf("文件头 (entry count): %d\n", headerVal)
	fmt.Printf("条目数: %d\n", len(entries))
	fmt.Printf("数据起始: 0x%X\n", dataStart)

	extCounts := map[string]int{}
	extSizes := map[string]int64{}
	for _, e := range entries {
		ext := "(none)"
		if idx := strings.LastIndex(e.Name, "."); idx >= 0 {
			ext = e.Name[idx+1:]
		}
		extCounts[ext]++
		extSizes[ext] += int64(e.Size)
	}

	type extInfo struct {
		ext   string
		count int
		size  int64
	}
	var exts []extInfo
	for ext, count := range extCounts {
		exts = append(exts, extInfo{ext, count, extSizes[ext]})
	}
	sort.Slice(exts, func(i, j int) bool { return exts[i].count > exts[j].count })

	fmt.Println("\n文件类型:")
	for _, e := range exts {
		fmt.Printf("  .%s: %d 个文件 (%.1f MB)\n", e.ext, e.count, float64(e.size)/1024/1024)
	}

	fmt.Println("\nCSV 文本文件:")
	for _, e := range entries {
		if strings.HasSuffix(e.Name, ".csv") {
			fmt.Printf("  %-50s %10d 字节\n", e.Name, e.Size)
		}
	}
}

func cmdExtract() {
	header, entries, _, err := parseIndex(gpakPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: %v\n", err)
		os.Exit(1)
	}

	os.MkdirAll(extractDir, 0755)

	// 保存文件头
	headerPath := filepath.Join(extractDir, "__gpak_header.bin")
	os.WriteFile(headerPath, header, 0644)

	// 保存索引
	indexPath := filepath.Join(extractDir, "__gpak_index.txt")
	var indexLines []string
	for _, e := range entries {
		indexLines = append(indexLines, e.Name)
	}
	os.WriteFile(indexPath, []byte(strings.Join(indexLines, "\n")+"\n"), 0644)

	fmt.Printf("解包 %d 个文件...\n", len(entries))

	f, err := os.Open(gpakPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	buf := make([]byte, 8*1024*1024)
	for i, e := range entries {
		outPath := filepath.Join(extractDir, filepath.FromSlash(e.Name))
		os.MkdirAll(filepath.Dir(outPath), 0755)

		f.Seek(e.Offset, io.SeekStart)
		out, err := os.Create(outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  创建文件失败 %s: %v\n", e.Name, err)
			continue
		}

		remaining := int64(e.Size)
		for remaining > 0 {
			n := int64(len(buf))
			if n > remaining {
				n = remaining
			}
			nr, err := f.Read(buf[:n])
			if err != nil || nr == 0 {
				break
			}
			out.Write(buf[:nr])
			remaining -= int64(nr)
		}
		out.Close()

		if (i+1)%200 == 0 || i == len(entries)-1 {
			fmt.Printf("  [%d/%d] %s\n", i+1, len(entries), e.Name)
		}
	}
	fmt.Printf("完成。已解包到 %s/\n", extractDir)
}

func cmdPack() {
	headerPath := filepath.Join(extractDir, "__gpak_header.bin")
	indexPath := filepath.Join(extractDir, "__gpak_index.txt")

	header, err := os.ReadFile(headerPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: 找不到 __gpak_header.bin，请先执行 extract\n")
		os.Exit(1)
	}

	indexData, err := os.ReadFile(indexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: 找不到 __gpak_index.txt，请先执行 extract\n")
		os.Exit(1)
	}

	var fileList []string
	for _, line := range strings.Split(string(indexData), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			fileList = append(fileList, line)
		}
	}

	fmt.Printf("打包 %d 个文件到 %s...\n", len(fileList), gpakPatched)

	// 构建原始文件映射（用于找不到本地文件时从原始 GPAK 读取）
	_, origEntries, _, err := parseIndex(gpakPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: 无法读取原始 %s: %v\n", gpakPath, err)
		os.Exit(1)
	}
	origMap := map[string]gpakEntry{}
	for _, e := range origEntries {
		origMap[e.Name] = e
	}

	type fileInfo struct {
		name       string
		size       uint32
		localPath  string // 非空 = 从本地读取
		origOffset int64  // localPath 为空时从原始 GPAK 读取
	}

	var files []fileInfo
	for _, name := range fileList {
		localPath := filepath.Join(extractDir, filepath.FromSlash(name))
		if fi, err := os.Stat(localPath); err == nil {
			files = append(files, fileInfo{name: name, size: uint32(fi.Size()), localPath: localPath})
		} else if orig, ok := origMap[name]; ok {
			files = append(files, fileInfo{name: name, size: orig.Size, origOffset: orig.Offset})
		} else {
			fmt.Fprintf(os.Stderr, "  警告: %s 既不在本地也不在原始 GPAK 中\n", name)
		}
	}

	// 写入打包文件
	out, err := os.Create(gpakPatched)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: 无法创建 %s: %v\n", gpakPatched, err)
		os.Exit(1)
	}
	defer out.Close()

	// 写文件头
	out.Write(header)

	// 写索引
	for _, fi := range files {
		nameBytes := []byte(fi.name)
		binary.Write(out, binary.LittleEndian, uint16(len(nameBytes)))
		out.Write(nameBytes)
		binary.Write(out, binary.LittleEndian, fi.size)
	}

	// 写文件数据
	origF, err := os.Open(gpakPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: 无法打开 %s: %v\n", gpakPath, err)
		os.Exit(1)
	}
	defer origF.Close()

	buf := make([]byte, 8*1024*1024)
	for i, fi := range files {
		if fi.localPath != "" {
			src, err := os.Open(fi.localPath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  读取失败 %s: %v\n", fi.name, err)
				continue
			}
			io.CopyBuffer(out, src, buf)
			src.Close()
		} else {
			origF.Seek(fi.origOffset, io.SeekStart)
			io.CopyBuffer(out, io.LimitReader(origF, int64(fi.size)), buf)
		}

		if (i+1)%500 == 0 || i == len(files)-1 {
			fmt.Printf("  [%d/%d] %s\n", i+1, len(files), fi.name)
		}
	}

	outFi, _ := os.Stat(gpakPatched)
	fmt.Printf("完成。输出: %s (%.1f MB)\n", gpakPatched, float64(outFi.Size())/1024/1024)
}

func cmdApply() {
	if _, err := os.Stat(gpakPatched); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "错误: 找不到 %s，请先执行 pack\n", gpakPatched)
		os.Exit(1)
	}

	if _, err := os.Stat(gpakBackup); os.IsNotExist(err) {
		fmt.Printf("备份 %s -> %s\n", gpakPath, gpakBackup)
		copyFile(gpakPath, gpakBackup)
	} else {
		fmt.Printf("备份已存在: %s\n", gpakBackup)
	}

	fmt.Printf("替换 %s\n", gpakPath)
	os.Remove(gpakPath)
	if err := os.Rename(gpakPatched, gpakPath); err != nil {
		// rename 可能跨卷失败，用复制
		copyFile(gpakPatched, gpakPath)
		os.Remove(gpakPatched)
	}
	fmt.Println("完成。补丁已应用。")
}

func copyFile(src, dst string) {
	s, err := os.Open(src)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: %v\n", err)
		os.Exit(1)
	}
	defer s.Close()

	d, err := os.Create(dst)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: %v\n", err)
		os.Exit(1)
	}
	defer d.Close()

	buf := make([]byte, 8*1024*1024)
	io.CopyBuffer(d, s, buf)
}
