"""Core modular components for Berlin Grid Digital Twin."""
import pandas as pd
import numpy as np
import time
try:
    import pandapower as pp
except ImportError:
    pp = None


class DataLayer:
    """Handles Berlin grid data loading and cleaning."""
    
    @staticmethod
    def clean_german_float(x):
        """Convert German float notation to Python float."""
        if pd.isna(x): 
            return 0.0
        if isinstance(x, str):
            clean = x.replace('.', '').replace(',', '.')
            try: 
                return float(clean)
            except: 
                return 0.0
        return float(x)
    
    @staticmethod
    def load_berlin_data(data_dir='data'):
        """Load and clean Berlin grid CSV data."""
        files_map = {
            'Gen_MS_kW': f'{data_dir}/MS_Erzeugung.csv',
            'Total_Load_MS_kW': f'{data_dir}/MS_Verbrauch.csv',
            'Grid_Import_MS_kW': f'{data_dir}/MS_Import.csv'
        }
        
        dfs = []
        for col_name, filepath in files_map.items():
            df_temp = pd.read_csv(filepath, sep=';', decimal=',', thousands='.')
            df_temp['Datetime'] = pd.to_datetime(df_temp.iloc[:, 0], dayfirst=True)
            df_temp[col_name] = df_temp.iloc[:, 1].apply(DataLayer.clean_german_float)
            dfs.append(df_temp[['Datetime', col_name]])
        
        df = dfs[0]
        for df_temp in dfs[1:]:
            df = df.merge(df_temp, on='Datetime', how='inner')
        
        df['Gen_MS_kW'] = df['Gen_MS_kW'].abs()
        df['Gen_MS_MW'] = df['Gen_MS_kW'] / 1000.0
        df['Total_Load_MS_MW'] = df['Total_Load_MS_kW'] / 1000.0
        df['Grid_Import_MS_MW'] = df['Grid_Import_MS_kW'] / 1000.0
        
        return df


class PhysicalTwin:
    """AC power flow simulation using Pandapower."""
    
    def __init__(self):
        self.net = self._build_model()
    
    def _build_model(self):
        """Create Berlin-like grid topology."""
        if pp is None:
            return None
        
        net = pp.create_empty_network()
        
        # External grid (substation)
        pp.create_bus(net, vn_kv=10.0, name="External_Grid")
        pp.create_ext_grid(net, bus=0, vm_pu=1.00, name="Substation")
        
        # MV feeder buses
        pp.create_bus(net, vn_kv=10.0, name="MV_Feeder_1")
        pp.create_line_from_parameters(
            net, from_bus=0, to_bus=1, length_km=2.0,
            r_ohm_per_km=0.16, x_ohm_per_km=0.10, c_nf_per_km=50,
            max_i_ka=0.42, name="Line_External_MV1"
        )
        
        pp.create_bus(net, vn_kv=10.0, name="MV_Feeder_2")
        pp.create_line_from_parameters(
            net, from_bus=0, to_bus=2, length_km=2.5,
            r_ohm_per_km=0.16, x_ohm_per_km=0.10, c_nf_per_km=50,
            max_i_ka=0.42, name="Line_External_MV2"
        )
        
        # LV buses via transformers
        pp.create_bus(net, vn_kv=0.4, name="LV_Bus_1")
        pp.create_transformer_from_parameters(
            net, hv_bus=1, lv_bus=3,
            sn_mva=0.63, vn_hv_kv=10.0, vn_lv_kv=0.4,
            vkr_percent=1.2, vk_percent=4.0, pfe_kw=1.2, i0_percent=0.4,
            name="Trafo_1"
        )
        
        pp.create_bus(net, vn_kv=0.4, name="LV_Bus_2")
        pp.create_transformer_from_parameters(
            net, hv_bus=2, lv_bus=4,
            sn_mva=0.63, vn_hv_kv=10.0, vn_lv_kv=0.4,
            vkr_percent=1.2, vk_percent=4.0, pfe_kw=1.2, i0_percent=0.4,
            name="Trafo_2"
        )
        
        # Loads and generation
        pp.create_load(net, bus=3, p_mw=0.2, q_mvar=0.05, name="Load_LV1")
        pp.create_load(net, bus=4, p_mw=0.3, q_mvar=0.08, name="Load_LV2")
        
        pp.create_sgen(net, bus=3, p_mw=0.15, q_mvar=0.0, name="PV_LV1")
        pp.create_sgen(net, bus=4, p_mw=0.20, q_mvar=0.0, name="PV_LV2")
        
        return net
    
    def run_power_flow(self, gen_mw, load_mw):
        """Execute AC power flow and return voltage/loading results."""
        if self.net is None:
            return None
        
        # Update loads and generation
        self.net.load.at[0, 'p_mw'] = load_mw * 0.4
        self.net.load.at[1, 'p_mw'] = load_mw * 0.6
        self.net.sgen.at[0, 'p_mw'] = gen_mw * 0.4
        self.net.sgen.at[1, 'p_mw'] = gen_mw * 0.6
        
        try:
            pp.runpp(self.net, algorithm='nr', numba=False)
            return {
                'voltage_min_pu': self.net.res_bus['vm_pu'].min(),
                'voltage_max_pu': self.net.res_bus['vm_pu'].max(),
                'line_loading_max_pct': self.net.res_line['loading_percent'].max(),
                'trafo_loading_max_pct': self.net.res_trafo['loading_percent'].max()
            }
        except Exception as e:
            return None


class Controller:
    """Fuzzy logic controller with sigmoid smoothing."""
    
    def __init__(self, limit_mw=100.0):
        self.limit = limit_mw
    
    def fuzzy_control(self, load_mw, k=15, s_ref=0.95):
        """Sigmoid-based fuzzy curtailment.
        
        Args:
            load_mw: Current load in MW
            k: Steepness factor (higher = sharper transition)
            s_ref: Threshold stress level (0-1)
        
        Returns:
            Curtailed load in MW
        """
        stress = load_mw / self.limit
        activation = 1 / (1 + np.exp(-k * (stress - s_ref)))
        dimming = 1 - (activation * 0.20)  # Max 20% curtailment
        return np.minimum(load_mw * dimming, self.limit)


class StochasticEngine:
    """AR(1) noise generation for Monte Carlo."""
    
    @staticmethod
    def generate_ar1_noise(n, sigma, phi):
        """Generate Auto-Regressive AR(1) noise.
        
        Args:
            n: Number of samples
            sigma: Standard deviation
            phi: Persistence factor (0=random, 0.95=high persistence)
        
        Returns:
            AR(1) noise array
        """
        noise = np.zeros(n)
        white_noise = np.random.normal(0, sigma * np.sqrt(1 - phi**2), n)
        
        for t in range(1, n):
            noise[t] = phi * noise[t-1] + white_noise[t]
        
        return noise
