import { Theme } from './settings/types';
import { SaaSNavbar } from './components/generated/SaaSNavbar';

let theme: Theme = 'light';

function App() {
  function setTheme(theme: Theme) {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }

  setTheme(theme);

  return (
    <>
      <SaaSNavbar />
    </>
  ); // %EXPORT_STATEMENT%
}

export default App;
