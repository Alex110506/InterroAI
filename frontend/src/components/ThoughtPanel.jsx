import s from './ThoughtPanel.module.css'

export default function ThoughtPanel({ isOpen }) {
  return (
    <div className={`${s.panel} ${isOpen ? '' : s.collapsed}`}>
      <div className={s.drag} />
      <div className={s.body} />
    </div>
  )
}
