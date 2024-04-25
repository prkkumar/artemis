#include "FerroE.H"
#include "Eff_Field_Landau.H"
#include "FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H"
#include "Utils/WarpXUtil.H"
#include "WarpX.H"
#include <ablastr/coarsen/sample.H>
#include "Utils/Parser/IntervalsParser.H"
#include "Utils/Parser/ParserUtils.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealVect.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Scan.H>

#include <AMReX_BaseFwd.H>

#include <memory>
#include <sstream>

FerroE::FerroE ()
{
    amrex::Print() << " FerroE class is constructed\n";
    ReadParameters();
}

void
FerroE::ReadParameters ()
{
    amrex::ParmParse pp_ferroe("ferroe");

    utils::parser::Store_parserString(pp_ferroe, "ferroelectric_function(x,y,z)", m_str_ferroelectric_function);
    m_ferroelectric_parser = std::make_unique<amrex::Parser>(
                                   utils::parser::makeParser(m_str_ferroelectric_function, {"x", "y", "z"}));
}

void
FerroE::InitData()
{
    auto& warpx = WarpX::GetInstance();

    const int lev = 0;
    amrex::BoxArray ba = warpx.boxArray(lev);
    amrex::DistributionMapping dmap = warpx.DistributionMap(lev);
    // number of guard cells used in EB solver
    const amrex::IntVect ng_EB_alloc = warpx.getngEB();
    // Define a nodal multifab to store if region is on super conductor (1) or not (0)
    const amrex::IntVect nodal_flag = amrex::IntVect::TheNodeVector();
    const int ncomps = 1;
    m_ferroelectric_mf = std::make_unique<amrex::MultiFab>(amrex::convert(ba,nodal_flag), dmap, ncomps, ng_EB_alloc);

    InitializeFerroelectricMultiFabUsingParser(m_ferroelectric_mf.get(), m_ferroelectric_parser->compile<3>(), lev);

}

void
FerroE::EvolveFerroEJ (amrex::Real dt)
{
    amrex::Print() << " evolve Ferroelectric J using P\n";

    auto & warpx = WarpX::GetInstance();
    const int lev = 0;
    const amrex::IntVect ng_EB_alloc = warpx.getngEB();

    amrex::MultiFab * jx = warpx.get_pointer_current_fp(lev, 0);
    amrex::MultiFab * jy = warpx.get_pointer_current_fp(lev, 1);
    amrex::MultiFab * jz = warpx.get_pointer_current_fp(lev, 2);

    //Px, Py, and Pz have 2 components each. Px(i,j,k,0) is Px and Px(i,j,k,1) is dPx/dt and so on..
    amrex::MultiFab * Px = warpx.get_pointer_polarization_fp(lev, 0);
    amrex::MultiFab * Py = warpx.get_pointer_polarization_fp(lev, 1);
    amrex::MultiFab * Pz = warpx.get_pointer_polarization_fp(lev, 2);

    // J_tot  = free electric current + polarization current (dP/dt)

    for (amrex::MFIter mfi(*jx, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        //Extract field data
        amrex::Array4<amrex::Real> const& jx_arr = jx->array(mfi);
        amrex::Array4<amrex::Real> const& jy_arr = jy->array(mfi);
        amrex::Array4<amrex::Real> const& jz_arr = jz->array(mfi);
        amrex::Array4<amrex::Real> const& dPx_dt_arr = Px->array(mfi);
        amrex::Array4<amrex::Real> const& dPy_dt_arr = Py->array(mfi);
        amrex::Array4<amrex::Real> const& dPz_dt_arr = Pz->array(mfi);
	amrex::Array4<amrex::Real> const& fe_arr = m_ferroelectric_mf->array(mfi);
        amrex::Box const& tjx = mfi.tilebox(jx->ixType().toIntVect());
        amrex::Box const& tjy = mfi.tilebox(jy->ixType().toIntVect());
        amrex::Box const& tjz = mfi.tilebox(jz->ixType().toIntVect());


    amrex::ParallelFor(tjx, tjy, tjz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i+1,j,k)==1) {
                jx_arr(i,j,k) += dPx_dt_arr(i,j,k,1);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i,j+1,k)==1) {
                jy_arr(i,j,k) += dPy_dt_arr(i,j,k,1);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i,j,k+1)==1) {
                jz_arr(i,j,k) += dPz_dt_arr(i,j,k,1);
            }
        }
    );
    }

}

void
FerroE::InitializeFerroelectricMultiFabUsingParser (
                       amrex::MultiFab *sc_mf,
                       amrex::ParserExecutor<3> const& sc_parser,
                       const int lev)
{
    using namespace amrex::literals;

    WarpX& warpx = WarpX::GetInstance();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx_lev = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    amrex::IntVect iv = sc_mf->ixType().toIntVect();
    for ( amrex::MFIter mfi(*sc_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {
        // Initialize ghost cells in addition to valid cells

        const amrex::Box& tb = mfi.tilebox( iv, sc_mf->nGrowVect());
        amrex::Array4<amrex::Real> const& sc_fab =  sc_mf->array(mfi);
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                // Shift x, y, z position based on index type (only 3D supported for now)
                amrex::Real fac_x = (1._rt - iv[0]) * dx_lev[0] * 0.5_rt;
                amrex::Real x = i * dx_lev[0] + real_box.lo(0) + fac_x;
                amrex::Real fac_y = (1._rt - iv[1]) * dx_lev[1] * 0.5_rt;
                amrex::Real y = j * dx_lev[1] + real_box.lo(1) + fac_y;
                amrex::Real fac_z = (1._rt - iv[2]) * dx_lev[2] * 0.5_rt;
                amrex::Real z = k * dx_lev[2] + real_box.lo(2) + fac_z;
                // initialize the macroparameter
                sc_fab(i,j,k) = sc_parser(x,y,z);
        });

    }
}


/*Evolution of the polarization P is governed by mu*d^2P/dt^2 + gamma*dP/dt = E_eff
 *Let dP/dt = v, then we need to solve the system of following two first-order ODEs
 *dv/dt = (E_eff - gamma*v)/mu
 *dP/dt = v
 */
AMREX_GPU_DEVICE
amrex::Real func(amrex::Real E_eff, amrex::Real v,  amrex::Real gamma, amrex::Real mu)
{
     return (E_eff - gamma*v) / mu;
}

// RK4 time integrator
AMREX_GPU_DEVICE
void update_v(amrex::Real& result, amrex::Real dt, amrex::Real E_eff, amrex::Real gamma, amrex::Real mu) 
{
        int use_RK4 = 1;

        amrex::Real k1, k2, k3, k4;
        
        // Calculate k1
        k1 = dt * func(E_eff, result, gamma, mu);
        
        // Calculate k2
        k2 = dt * func(E_eff, result + 0.5*k1, gamma, mu);
        
        // Calculate k3
        k3 = dt * func(E_eff, result + 0.5*k2, gamma, mu);
        
        // Calculate k4
        k4 = dt * func(E_eff, result + k3, gamma, mu);
       
        if (use_RK4){
           // Update result using weighted sum of ks
           result += (1.0 / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        } else {
           result += k1; 
        }
}

void
FerroE::EvolveP (amrex::Real dt)
{
     amrex::Print() << " evolve P \n";
     auto & warpx = WarpX::GetInstance();
     int include_Landau = warpx.include_Landau;
     const int lev = 0;

     //Px, Py, and Pz have 2 components each. Px(i,j,k,0) is Px and Px(i,j,k,1) is dPx/dt and so on..
     amrex::MultiFab * Px = warpx.get_pointer_polarization_fp(lev, 0);
     amrex::MultiFab * Py = warpx.get_pointer_polarization_fp(lev, 1);
     amrex::MultiFab * Pz = warpx.get_pointer_polarization_fp(lev, 2);

     amrex::MultiFab * Ex = warpx.get_pointer_Efield_fp(lev, 0);
     amrex::MultiFab * Ey = warpx.get_pointer_Efield_fp(lev, 1);
     amrex::MultiFab * Ez = warpx.get_pointer_Efield_fp(lev, 2);

    for (amrex::MFIter mfi(*Px, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        //Extract field data
        amrex::Array4<amrex::Real> const& Px_arr = Px->array(mfi);
        amrex::Array4<amrex::Real> const& Py_arr = Py->array(mfi);
        amrex::Array4<amrex::Real> const& Pz_arr = Pz->array(mfi);
        amrex::Array4<amrex::Real> const& Ex_arr = Ex->array(mfi);
        amrex::Array4<amrex::Real> const& Ey_arr = Ey->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Ez->array(mfi);
	amrex::Array4<amrex::Real> const& fe_arr = m_ferroelectric_mf->array(mfi);
        amrex::Box const& tpx = mfi.tilebox(Px->ixType().toIntVect());
        amrex::Box const& tpy = mfi.tilebox(Py->ixType().toIntVect());
        amrex::Box const& tpz = mfi.tilebox(Pz->ixType().toIntVect());


    amrex::ParallelFor(tpx, tpy, tpz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i+1,j,k)==1) {
	        
                amrex::Real Ex_eff = Ex_arr(i,j,k);

                if (include_Landau == 1){
                   Ex_eff += compute_ex_Landau(Px_arr(i,j,k,0), Py_arr(i,j,k,0), Pz_arr(i,j,k,0));
                }

                  //get dPx/dt using numerical integration
                  update_v(Px_arr(i,j,k,1), dt, Ex_eff, gamma, mu);
                  //get Px 
                  Px_arr(i,j,k,0) += dt*Px_arr(i,j,k,1);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i,j+1,k)==1) {

                amrex::Real Ey_eff = Ey_arr(i,j,k);

                if (include_Landau == 1){
                   Ey_eff += compute_ey_Landau(Px_arr(i,j,k,0), Py_arr(i,j,k,0), Pz_arr(i,j,k,0));
                }
                  
                  //get dPy/dt using numerical integration
                  update_v(Py_arr(i,j,k,1), dt, Ey_eff, gamma, mu);
                  //get Py 
                  Py_arr(i,j,k,0) += dt*Py_arr(i,j,k,1);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i,j,k+1)==1) {

                amrex::Real Ez_eff = Ez_arr(i,j,k);

                if (include_Landau == 1){
                   Ez_eff += compute_ez_Landau(Px_arr(i,j,k,0), Py_arr(i,j,k,0), Pz_arr(i,j,k,0));
                }
                  
                  //get dPz/dt using numerical integration
                  update_v(Pz_arr(i,j,k,1), dt, Ez_eff, gamma, mu);
                  //get Pz 
                  Pz_arr(i,j,k,0) += dt*Pz_arr(i,j,k,1);
            }
        }
    );
    }
}

