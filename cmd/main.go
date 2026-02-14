package main

import (
	"fmt"
	"os"
	"time"
)

func main() {
	cmd := "patch"
	if len(os.Args) >= 2 {
		cmd = os.Args[1]
	}

	switch cmd {
	case "patch":
		cmdPatch()
		waitExit()
	case "extract":
		cmdExtract()
	case "apply-translations":
		cmdApplyTranslations()
	case "replace-font":
		cmdReplaceFont()
	case "pack":
		cmdPack()
	case "apply":
		cmdApply()
	case "info":
		cmdInfo()
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "未知命令: %s\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Mewgenics 简体中文汉化补丁工具")
	fmt.Println()
	fmt.Println("用法: mewpatch <命令>")
	fmt.Println()
	fmt.Println("命令:")
	fmt.Println("  patch              一键执行全部步骤（解包→翻译→字体→打包→替换）")
	fmt.Println("  extract            解包 resources.gpak")
	fmt.Println("  apply-translations 应用翻译到 CSV 文件")
	fmt.Println("  replace-font       替换 SWF 中的中文字体")
	fmt.Println("  pack               重新打包为 resources_patched.gpak")
	fmt.Println("  apply              用补丁文件替换 resources.gpak")
	fmt.Println("  info               查看 GPAK 文件信息")
}

func cmdPatch() {
	start := time.Now()
	fmt.Println("=== Mewgenics 汉化补丁 ===")
	fmt.Println()

	fmt.Println("[1/5] 解包资源文件...")
	cmdExtract()
	fmt.Println()

	fmt.Println("[2/5] 应用翻译...")
	cmdApplyTranslations()
	fmt.Println()

	fmt.Println("[3/5] 替换中文字体...")
	cmdReplaceFont()
	fmt.Println()

	fmt.Println("[4/5] 重新打包...")
	cmdPack()
	fmt.Println()

	fmt.Println("[5/5] 应用补丁...")
	cmdApply()
	fmt.Println()

	fmt.Println("清理临时文件...")
	os.RemoveAll(extractDir)
	fmt.Println()

	fmt.Printf("✓ 全部完成！耗时 %.1f 秒\n", time.Since(start).Seconds())
}

func waitExit() {
	fmt.Println()
	fmt.Println("按回车键退出...")
	fmt.Scanln()
}
