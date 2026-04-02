import { createContext, useContext, useState } from 'react'
import { translations } from './translations'

const LocaleContext = createContext()

export function LocaleProvider({ children }) {
  const [locale, setLocale] = useState('vi')

  function toggle() {
    setLocale(prev => prev === 'vi' ? 'en' : 'vi')
  }

  return (
    <LocaleContext.Provider value={{ locale, toggle, t: translations[locale] }}>
      {children}
    </LocaleContext.Provider>
  )
}

export function useLocale() {
  return useContext(LocaleContext)
}
