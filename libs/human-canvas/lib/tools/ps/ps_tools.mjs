// dsh-ps / lib/ps_tools.mjs — 用 ExtendScript(ActionManager) 封装 PS 的"CV 做不到"操作
import { cmd } from './ps_session.mjs'

export function openImage(path) {
  return cmd('var f=new File(' + JSON.stringify(path) + '); var d=app.open(f); (Number(d.width))+"x"+(Number(d.height));')
}

// 背景层转普通图层（这样"清除选区"才是变透明，而非填背景色）
export function toLayer() {
  return cmd('var d=app.activeDocument; if(d.activeLayer.isBackgroundLayer){ d.activeLayer.isBackgroundLayer=false; } "toLayer";')
}

// 魔棒式"点背景选中整片背景"：在(x,y)画 2x2 微选区，再 selection.similar() 扩到所有相似色。
// 用纯 DOM 方法（Select > Similar），规避 ActionManager 在 CS6 的脆弱性。
export function selectSimilarAt(x, y) {
  const js = 'var d=app.activeDocument; var s=d.selection; s.deselect();' +
    's.select([[' + x + ',' + y + '],[' + (x + 3) + ',' + y + '],[' + (x + 3) + ',' + (y + 3) + '],[' + x + ',' + (y + 3) + ']]);' +
    's.similar(); "similar:"+s.bounds.join(",");'
  return cmd(js)
}

// 删除(清除)当前选区像素（背景变透明）
export function clearSelection() {
  return cmd('executeAction(app.charIDToTypeID("clea"), undefined, DialogModes.NO); "cleared";')
}

export function selectionBounds() {
  return cmd('app.activeDocument.selection.bounds.join(",");')
}

// 导出为带透明背景的 PNG
export function exportPng(path) {
  const js = 'var d=app.activeDocument; d.saveAs(new File(' + JSON.stringify(path) +
    '), new PNGFormat(), true, Extension.LOWERCASE); "saved";'
  return cmd(js)
}
