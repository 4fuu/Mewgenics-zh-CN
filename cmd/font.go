package main

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"

	mewpatch "github.com/4fuu/Mewgenics-zh-CN"

	"golang.org/x/image/font"
	"golang.org/x/image/font/sfnt"
	"golang.org/x/image/math/fixed"
)

const (
	unicodeSWF = "extracted/swfs/unicodefont.swf"
	swfEM      = 1024 * 20 // 20480 twips
)

// ---------- bit writer ----------

type bitWriter struct {
	bits []byte // 每个元素为 0 或 1
}

func (bw *bitWriter) writeUB(nbits int, value uint32) {
	for i := nbits - 1; i >= 0; i-- {
		bw.bits = append(bw.bits, byte((value>>uint(i))&1))
	}
}

func (bw *bitWriter) writeSB(nbits int, value int32) {
	var v uint32
	if value < 0 {
		v = uint32((int64(1) << uint(nbits)) + int64(value))
	} else {
		v = uint32(value)
	}
	bw.writeUB(nbits, v)
}

func (bw *bitWriter) align() {
	for len(bw.bits)%8 != 0 {
		bw.bits = append(bw.bits, 0)
	}
}

func (bw *bitWriter) toBytes() []byte {
	bw.align()
	result := make([]byte, len(bw.bits)/8)
	for i := 0; i < len(bw.bits); i += 8 {
		var b byte
		for j := 0; j < 8; j++ {
			b = b << 1
			if i+j < len(bw.bits) {
				b |= bw.bits[i+j]
			}
		}
		result[i/8] = b
	}
	return result
}

func bitsNeededSigned(value int32) int {
	if value == 0 {
		return 1
	}
	if value > 0 {
		return bitLen32(uint32(value)) + 1
	}
	neg := -value - 1
	if value == -1 {
		neg = 0
	}
	return bitLen32(uint32(neg)) + 2
}

func bitLen32(v uint32) int {
	n := 0
	for v > 0 {
		n++
		v >>= 1
	}
	return n
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func scale26_6(v fixed.Int26_6, scale float64) int32 {
	return int32(math.Round((float64(v) / 64.0) * scale))
}

// ---------- SWF shape encoding ----------

type swfSegment struct {
	op   int // 0=moveTo, 1=lineTo, 2=quadTo
	x, y int32
	// quadTo 专用
	cx, cy int32
}

type swfContour struct {
	segments []swfSegment
}

func encodeShape(contours []swfContour) []byte {
	bw := &bitWriter{}
	bw.writeUB(4, 1) // NumFillBits = 1
	bw.writeUB(4, 0) // NumLineBits = 0

	for _, contour := range contours {
		if len(contour.segments) == 0 {
			continue
		}

		first := contour.segments[0]
		curX, curY := first.x, first.y

		moveBits := maxInt(bitsNeededSigned(curX), bitsNeededSigned(curY))
		if moveBits < 1 {
			moveBits = 1
		}

		// StyleChangeRecord: MoveTo + FillStyle1
		bw.writeUB(1, 0) // TypeFlag = 0
		bw.writeUB(1, 0) // StateNewStyles
		bw.writeUB(1, 0) // StateLineStyle
		bw.writeUB(1, 1) // StateFillStyle1
		bw.writeUB(1, 0) // StateFillStyle0
		bw.writeUB(1, 1) // StateMoveTo
		bw.writeUB(5, uint32(moveBits))
		bw.writeSB(moveBits, curX)
		bw.writeSB(moveBits, curY)
		bw.writeUB(1, 1) // FillStyle1 = 1

		for _, seg := range contour.segments[1:] {
			switch seg.op {
			case 1: // lineTo
				dx := seg.x - curX
				dy := seg.y - curY
				if dx == 0 && dy == 0 {
					continue
				}
				nbits := maxInt(bitsNeededSigned(dx), bitsNeededSigned(dy))
				if nbits < 2 {
					nbits = 2
				}
				bw.writeUB(1, 1) // TypeFlag = 1
				bw.writeUB(1, 1) // StraightFlag
				bw.writeUB(4, uint32(nbits-2))
				if dx == 0 {
					bw.writeUB(1, 0) // GeneralLine = 0
					bw.writeUB(1, 1) // VertLine
					bw.writeSB(nbits, dy)
				} else if dy == 0 {
					bw.writeUB(1, 0) // GeneralLine = 0
					bw.writeUB(1, 0) // HorizLine
					bw.writeSB(nbits, dx)
				} else {
					bw.writeUB(1, 1) // GeneralLine
					bw.writeSB(nbits, dx)
					bw.writeSB(nbits, dy)
				}
				curX, curY = seg.x, seg.y

			case 2: // quadTo
				cdx := seg.cx - curX
				cdy := seg.cy - curY
				adx := seg.x - seg.cx
				ady := seg.y - seg.cy
				nbits := maxInt(
					maxInt(bitsNeededSigned(cdx), bitsNeededSigned(cdy)),
					maxInt(bitsNeededSigned(adx), bitsNeededSigned(ady)),
				)
				if nbits < 2 {
					nbits = 2
				}
				bw.writeUB(1, 1) // TypeFlag = 1
				bw.writeUB(1, 0) // StraightFlag = 0 (curve)
				bw.writeUB(4, uint32(nbits-2))
				bw.writeSB(nbits, cdx)
				bw.writeSB(nbits, cdy)
				bw.writeSB(nbits, adx)
				bw.writeSB(nbits, ady)
				curX, curY = seg.x, seg.y
			}
		}
	}

	// EndShapeRecord
	bw.writeUB(6, 0)
	return bw.toBytes()
}

func emptyShape() []byte {
	bw := &bitWriter{}
	bw.writeUB(4, 1) // NumFillBits
	bw.writeUB(4, 0) // NumLineBits
	bw.writeUB(6, 0) // EndShapeRecord
	return bw.toBytes()
}

// ---------- TTF → SWF glyph conversion ----------

func convertTTFGlyphs(ttfData []byte) (codes []uint16, shapesData [][]byte, advances []int16, ascent, descent uint16, err error) {
	f, err := sfnt.Parse(ttfData)
	if err != nil {
		return nil, nil, nil, 0, 0, fmt.Errorf("解析 TTF 失败: %w", err)
	}

	upm := f.UnitsPerEm()
	scale := float64(swfEM) / float64(upm)
	fmt.Printf("  unitsPerEm: %d, scale: %.4f\n", upm, scale)

	var buf sfnt.Buffer
	ppem := fixed.I(int(upm))
	metrics, err := f.Metrics(&buf, ppem, font.HintingNone)
	if err != nil {
		return nil, nil, nil, 0, 0, fmt.Errorf("读取字体度量失败: %w", err)
	}

	ascentVal := int(scale26_6(metrics.Ascent, scale))
	if ascentVal < 0 {
		ascentVal = -ascentVal
	}
	if ascentVal > 65535 {
		ascentVal = 65535
	}
	descentVal := int(scale26_6(metrics.Descent, scale))
	if descentVal < 0 {
		descentVal = -descentVal
	}
	if descentVal > 65535 {
		descentVal = 65535
	}
	ascent = uint16(ascentVal)
	descent = uint16(descentVal)
	fmt.Printf("  ascent: %d, descent: %d\n", ascent, descent)

	failed := 0
	for cp := rune(0); cp <= 0xFFFF; cp++ {
		gi, err := f.GlyphIndex(&buf, cp)
		if err != nil || gi == 0 {
			continue
		}

		// 获取 advance
		adv, err := f.GlyphAdvance(&buf, gi, ppem, font.HintingNone)
		if err != nil {
			failed++
			continue
		}
		advSwf := int16(scale26_6(adv, scale))

		// 加载轮廓
		segments, err := f.LoadGlyph(&buf, gi, ppem, nil)
		if err != nil {
			// 空字形
			codes = append(codes, uint16(cp))
			shapesData = append(shapesData, emptyShape())
			advances = append(advances, advSwf)
			continue
		}

		contours := segmentsToContours(segments, scale)
		var shapeBytes []byte
		if len(contours) == 0 {
			shapeBytes = emptyShape()
		} else {
			shapeBytes = encodeShape(contours)
		}

		codes = append(codes, uint16(cp))
		shapesData = append(shapesData, shapeBytes)
		advances = append(advances, advSwf)
	}

	fmt.Printf("  转换了 %d 个字形（%d 个失败）\n", len(codes), failed)
	return codes, shapesData, advances, ascent, descent, nil
}

func segmentsToContours(segments sfnt.Segments, scale float64) []swfContour {
	var contours []swfContour
	var cur *swfContour
	var startX, startY int32

	closeContour := func() {
		if cur == nil || len(cur.segments) == 0 {
			return
		}
		// 获取当前结束坐标
		last := cur.segments[len(cur.segments)-1]
		var endX, endY int32
		switch last.op {
		case 0, 1:
			endX, endY = last.x, last.y
		case 2:
			endX, endY = last.x, last.y
		}
		if endX != startX || endY != startY {
			cur.segments = append(cur.segments, swfSegment{op: 1, x: startX, y: startY})
		}
		contours = append(contours, *cur)
	}

	for _, seg := range segments {
		switch seg.Op {
		case sfnt.SegmentOpMoveTo:
			closeContour()
			cur = &swfContour{}
			x := scale26_6(seg.Args[0].X, scale)
			y := scale26_6(seg.Args[0].Y, scale)
			startX, startY = x, y
			cur.segments = append(cur.segments, swfSegment{op: 0, x: x, y: y})

		case sfnt.SegmentOpLineTo:
			if cur == nil {
				continue
			}
			x := scale26_6(seg.Args[0].X, scale)
			y := scale26_6(seg.Args[0].Y, scale)
			cur.segments = append(cur.segments, swfSegment{op: 1, x: x, y: y})

		case sfnt.SegmentOpQuadTo:
			if cur == nil {
				continue
			}
			cx := scale26_6(seg.Args[0].X, scale)
			cy := scale26_6(seg.Args[0].Y, scale)
			x := scale26_6(seg.Args[1].X, scale)
			y := scale26_6(seg.Args[1].Y, scale)
			cur.segments = append(cur.segments, swfSegment{op: 2, x: x, y: y, cx: cx, cy: cy})

		case sfnt.SegmentOpCubeTo:
			if cur == nil {
				continue
			}
			x := scale26_6(seg.Args[2].X, scale)
			y := scale26_6(seg.Args[2].Y, scale)
			cur.segments = append(cur.segments, swfSegment{op: 1, x: x, y: y})
		}
	}
	closeContour()
	return contours
}

// ---------- DefineFont3 解析 ----------

// parseDefineFont3Glyphs 从 DefineFont3 标签数据中提取字形信息
func parseDefineFont3Glyphs(data []byte) (codes []uint16, shapesData [][]byte, advances []int16, ascent, descent uint16, err error) {
	if len(data) < 6 {
		return nil, nil, nil, 0, 0, fmt.Errorf("DefineFont3 数据太短")
	}

	// FontID: U16
	// flags: U8
	// languageCode: U8
	// fontNameLen: U8
	flags := data[2]
	hasLayout := flags&0x80 != 0
	wideOffsets := flags&0x08 != 0
	// wideCodes := flags&0x04 != 0 // DefineFont3 始终使用 WideCodes

	fontNameLen := int(data[4])
	pos := 5 + fontNameLen

	if pos+2 > len(data) {
		return nil, nil, nil, 0, 0, fmt.Errorf("DefineFont3 数据截断（numGlyphs 处）")
	}
	numGlyphs := int(binary.LittleEndian.Uint16(data[pos:]))
	pos += 2

	if numGlyphs == 0 {
		return nil, nil, nil, 0, 0, nil
	}

	// 读取偏移表 (numGlyphs+1 个偏移)
	offsets := make([]int, numGlyphs+1)
	if wideOffsets {
		needed := (numGlyphs + 1) * 4
		if pos+needed > len(data) {
			return nil, nil, nil, 0, 0, fmt.Errorf("DefineFont3 数据截断（偏移表处）")
		}
		for i := 0; i <= numGlyphs; i++ {
			offsets[i] = int(binary.LittleEndian.Uint32(data[pos:]))
			pos += 4
		}
	} else {
		needed := (numGlyphs + 1) * 2
		if pos+needed > len(data) {
			return nil, nil, nil, 0, 0, fmt.Errorf("DefineFont3 数据截断（偏移表处）")
		}
		for i := 0; i <= numGlyphs; i++ {
			offsets[i] = int(binary.LittleEndian.Uint16(data[pos:]))
			pos += 2
		}
	}

	// shape 数据的基准位置 = 偏移表起始位置
	shapeBase := pos - offsets[0]

	// 提取每个字形的 shape 数据
	shapesData = make([][]byte, numGlyphs)
	for i := 0; i < numGlyphs; i++ {
		start := shapeBase + offsets[i]
		end := shapeBase + offsets[i+1]
		if start < 0 || end > len(data) || start > end {
			shapesData[i] = emptyShape()
			continue
		}
		shapeSlice := make([]byte, end-start)
		copy(shapeSlice, data[start:end])
		shapesData[i] = shapeSlice
	}

	// 跳到 code table 的位置
	pos = shapeBase + offsets[numGlyphs]

	// 读取 CodeTable: numGlyphs x U16
	if pos+numGlyphs*2 > len(data) {
		return nil, nil, nil, 0, 0, fmt.Errorf("DefineFont3 数据截断（CodeTable 处）")
	}
	codes = make([]uint16, numGlyphs)
	for i := 0; i < numGlyphs; i++ {
		codes[i] = binary.LittleEndian.Uint16(data[pos:])
		pos += 2
	}

	// 读取 Layout 信息
	if hasLayout {
		if pos+4 > len(data) {
			return codes, shapesData, nil, 0, 0, nil
		}
		ascent = binary.LittleEndian.Uint16(data[pos:])
		pos += 2
		descent = binary.LittleEndian.Uint16(data[pos:])
		pos += 2
		pos += 2 // leading (S16)

		// AdvanceTable: numGlyphs x S16
		if pos+numGlyphs*2 > len(data) {
			return codes, shapesData, nil, ascent, descent, nil
		}
		advances = make([]int16, numGlyphs)
		for i := 0; i < numGlyphs; i++ {
			advances[i] = int16(binary.LittleEndian.Uint16(data[pos:]))
			pos += 2
		}

		// 跳过 BoundsTable: numGlyphs x RECT
		for i := 0; i < numGlyphs; i++ {
			if pos >= len(data) {
				break
			}
			nbits := int(data[pos] >> 3)
			var rectSize int
			if nbits == 0 {
				rectSize = 1
			} else {
				rectSize = (5 + nbits*4 + 7) / 8
			}
			pos += rectSize
		}
		// KerningCount: U16 (不需要读取)
	}

	return codes, shapesData, advances, ascent, descent, nil
}

// ---------- 字形合并 ----------

type glyphEntry struct {
	shape   []byte
	advance int16
}

// mergeGlyphs 合并原始字形和新字形，新字形覆盖原始字形
func mergeGlyphs(origCodes []uint16, origShapes [][]byte, origAdvances []int16,
	newCodes []uint16, newShapes [][]byte, newAdvances []int16) ([]uint16, [][]byte, []int16) {

	m := make(map[uint16]glyphEntry)

	// 先填入原始字形
	for i, c := range origCodes {
		var adv int16
		if i < len(origAdvances) {
			adv = origAdvances[i]
		}
		m[c] = glyphEntry{shape: origShapes[i], advance: adv}
	}

	// 用新字形覆盖
	for i, c := range newCodes {
		var adv int16
		if i < len(newAdvances) {
			adv = newAdvances[i]
		}
		m[c] = glyphEntry{shape: newShapes[i], advance: adv}
	}

	// 按 codepoint 排序
	sortedCodes := make([]uint16, 0, len(m))
	for c := range m {
		sortedCodes = append(sortedCodes, c)
	}
	sort.Slice(sortedCodes, func(i, j int) bool { return sortedCodes[i] < sortedCodes[j] })

	mergedCodes := make([]uint16, len(sortedCodes))
	mergedShapes := make([][]byte, len(sortedCodes))
	mergedAdvances := make([]int16, len(sortedCodes))
	for i, c := range sortedCodes {
		e := m[c]
		mergedCodes[i] = c
		mergedShapes[i] = e.shape
		mergedAdvances[i] = e.advance
	}

	return mergedCodes, mergedShapes, mergedAdvances
}

// ---------- SWF DefineFont3 building ----------

func buildDefineFont3(fontID uint16, fontNameBytes []byte, codes []uint16, shapesData [][]byte, advances []int16, ascent, descent uint16) []byte {
	numGlyphs := len(codes)

	flags := byte(0x80 | 0x04) // HasLayout | WideCodes
	totalShape := 0
	for _, s := range shapesData {
		totalShape += len(s)
	}
	if totalShape+(numGlyphs+1)*4 > 65535 {
		flags |= 0x08 // WideOffsets
	}
	wideOffsets := flags&0x08 != 0

	var buf bytes.Buffer

	binary.Write(&buf, binary.LittleEndian, fontID)
	buf.WriteByte(flags)
	buf.WriteByte(0) // language code
	buf.WriteByte(byte(len(fontNameBytes)))
	buf.Write(fontNameBytes)
	binary.Write(&buf, binary.LittleEndian, uint16(numGlyphs))

	offsetTableSize := 0
	if wideOffsets {
		offsetTableSize = (numGlyphs + 1) * 4
	} else {
		offsetTableSize = (numGlyphs + 1) * 2
	}

	running := offsetTableSize
	var offsets []int
	for _, s := range shapesData {
		offsets = append(offsets, running)
		running += len(s)
	}
	offsets = append(offsets, running) // code_table_offset

	if wideOffsets {
		for _, o := range offsets {
			binary.Write(&buf, binary.LittleEndian, uint32(o))
		}
	} else {
		for _, o := range offsets {
			binary.Write(&buf, binary.LittleEndian, uint16(o))
		}
	}

	for _, s := range shapesData {
		buf.Write(s)
	}

	for _, c := range codes {
		binary.Write(&buf, binary.LittleEndian, c)
	}

	// Layout
	binary.Write(&buf, binary.LittleEndian, ascent)
	binary.Write(&buf, binary.LittleEndian, descent)
	binary.Write(&buf, binary.LittleEndian, int16(0)) // leading
	for _, a := range advances {
		binary.Write(&buf, binary.LittleEndian, a)
	}
	for i := 0; i < numGlyphs; i++ {
		buf.WriteByte(0x00) // empty RECT bounds
	}
	binary.Write(&buf, binary.LittleEndian, uint16(0)) // kerning count

	return buf.Bytes()
}

func buildTag(tagType uint16, data []byte) []byte {
	var buf bytes.Buffer
	dataLen := len(data)
	if dataLen < 0x3F {
		tc := (tagType << 6) | uint16(dataLen)
		binary.Write(&buf, binary.LittleEndian, tc)
		buf.Write(data)
	} else {
		tc := (tagType << 6) | 0x3F
		binary.Write(&buf, binary.LittleEndian, tc)
		binary.Write(&buf, binary.LittleEndian, uint32(dataLen))
		buf.Write(data)
	}
	return buf.Bytes()
}

func buildDefineFontName(fontID uint16, fullName string) []byte {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, fontID)
	buf.Write([]byte(fullName))
	buf.WriteByte(0)
	buf.WriteByte(0) // copyright (empty)
	return buf.Bytes()
}

func buildFontAlignZones(fontID uint16, numGlyphs int) []byte {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, fontID)
	buf.WriteByte(0) // CSM hint = thin
	for i := 0; i < numGlyphs; i++ {
		buf.WriteByte(2) // NumZoneData = 2
		binary.Write(&buf, binary.LittleEndian, uint16(0))
		binary.Write(&buf, binary.LittleEndian, uint16(0))
		binary.Write(&buf, binary.LittleEndian, uint16(0))
		binary.Write(&buf, binary.LittleEndian, uint16(0))
		buf.WriteByte(0x03) // ZoneMaskY | ZoneMaskX
	}
	return buf.Bytes()
}

// ---------- SWF tag parsing ----------

type swfTag struct {
	tagType   uint16
	offset    int
	headerLen int
	dataLen   int
}

func parseSWFTags(body []byte, headerSize int) []swfTag {
	var tags []swfTag
	off := headerSize
	for off < len(body)-1 {
		tc := binary.LittleEndian.Uint16(body[off:])
		tagType := tc >> 6
		tagLength := int(tc & 0x3F)
		hdrLen := 2
		if tagLength == 0x3F {
			tagLength = int(binary.LittleEndian.Uint32(body[off+2:]))
			hdrLen = 6
		}
		tags = append(tags, swfTag{tagType: tagType, offset: off, headerLen: hdrLen, dataLen: tagLength})
		off += hdrLen + tagLength
		if tagType == 0 {
			break
		}
	}
	return tags
}

// ---------- main entry ----------

func cmdReplaceFont() {
	swfPath := filepath.FromSlash(unicodeSWF)
	if _, err := os.Stat(swfPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "错误: 找不到 %s，请先执行 extract\n", unicodeSWF)
		os.Exit(1)
	}

	// 使用内嵌的原始 SWF 获取原始字形
	fmt.Println("解析内嵌原始字体...")
	origData := mewpatch.OriginalFontSWF

	sig := string(origData[:3])
	origVer := origData[3]

	var body []byte
	switch sig {
	case "CWS":
		r, err := zlib.NewReader(bytes.NewReader(origData[8:]))
		if err != nil {
			fmt.Fprintf(os.Stderr, "错误: 解压内嵌 SWF 失败: %v\n", err)
			os.Exit(1)
		}
		body, err = io.ReadAll(r)
		r.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "错误: 读取解压数据失败: %v\n", err)
			os.Exit(1)
		}
	case "FWS":
		body = origData[8:]
	default:
		fmt.Fprintf(os.Stderr, "错误: 未知 SWF 签名: %s\n", sig)
		os.Exit(1)
	}

	// 解析 RECT 大小
	nbits := int(body[0] >> 3)
	rectSize := (5 + nbits*4 + 7) / 8
	headerSize := rectSize + 4 // RECT + frame_rate(2) + frame_count(2)

	tags := parseSWFTags(body, headerSize)

	// 找到原始字体 ID、名称，并提取原始字形
	var origFontID uint16
	var origFontName []byte
	var origCodes []uint16
	var origShapes [][]byte
	var origAdvances []int16
	for _, tag := range tags {
		if tag.tagType == 75 { // DefineFont3
			data := body[tag.offset+tag.headerLen : tag.offset+tag.headerLen+tag.dataLen]
			origFontID = binary.LittleEndian.Uint16(data[0:2])
			nlen := int(data[4])
			origFontName = data[5 : 5+nlen]
			fontNameStr := string(origFontName)
			fmt.Printf("  原始字体: ID=%d, name=\"%s\"\n", origFontID, fontNameStr)

			// 解析原始字形
			var err error
			origCodes, origShapes, origAdvances, _, _, err = parseDefineFont3Glyphs(data)
			if err != nil {
				fmt.Fprintf(os.Stderr, "错误: 解析原始字形失败: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("  原始字形数: %d\n", len(origCodes))
			break
		}
	}

	if origFontName == nil {
		fmt.Fprintln(os.Stderr, "错误: 未在 SWF 中找到 DefineFont3")
		os.Exit(1)
	}

	// 转换中文 TTF 字形
	fmt.Println("转换 TTF 字形...")
	newCodes, newShapes, newAdvances, ascent, descent, err := convertTTFGlyphs(mewpatch.FontTTF)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: %v\n", err)
		os.Exit(1)
	}

	// 合并字形：中文字体优先，原始字形作为后备
	fmt.Println("合并字形...")
	newCodeSet := make(map[uint16]bool, len(newCodes))
	for _, c := range newCodes {
		newCodeSet[c] = true
	}
	origOnly := 0
	for _, c := range origCodes {
		if !newCodeSet[c] {
			origOnly++
		}
	}
	fmt.Printf("  中文字体字形: %d, 原始后备字形: %d\n", len(newCodes), origOnly)

	codes, shapesData, advances := mergeGlyphs(origCodes, origShapes, origAdvances, newCodes, newShapes, newAdvances)
	fmt.Printf("  合并后总字形数: %d\n", len(codes))

	// 构建新标签
	font3Data := buildDefineFont3(origFontID, origFontName, codes, shapesData, advances, ascent, descent)
	fmt.Printf("  DefineFont3 标签: %d 字节\n", len(font3Data))

	alignZonesData := buildFontAlignZones(origFontID, len(codes))
	fontNameData := buildDefineFontName(origFontID, string(origFontName))

	// 构建新 SWF
	fmt.Println("构建新 SWF...")
	var newBody bytes.Buffer
	newBody.Write(body[:headerSize])

	for _, tag := range tags {
		switch tag.tagType {
		case 75: // DefineFont3
			newBody.Write(buildTag(75, font3Data))
		case 73: // DefineFontAlignZones
			newBody.Write(buildTag(73, alignZonesData))
		case 88: // DefineFontName
			newBody.Write(buildTag(88, fontNameData))
		default:
			newBody.Write(body[tag.offset : tag.offset+tag.headerLen+tag.dataLen])
		}
	}

	// 写入 SWF（不压缩）
	fileLength := uint32(newBody.Len() + 8)
	var out bytes.Buffer
	out.Write([]byte("FWS"))
	out.WriteByte(origVer)
	binary.Write(&out, binary.LittleEndian, fileLength)
	out.Write(newBody.Bytes())

	if err := os.WriteFile(swfPath, out.Bytes(), 0644); err != nil {
		fmt.Fprintf(os.Stderr, "错误: 写入 SWF 失败: %v\n", err)
		os.Exit(1)
	}

	newSize := out.Len()
	fmt.Printf("完成。已写入 %s\n", unicodeSWF)
	fmt.Printf("  大小: %.1f MB, 字形数: %d\n", float64(newSize)/1024/1024, len(codes))
}
