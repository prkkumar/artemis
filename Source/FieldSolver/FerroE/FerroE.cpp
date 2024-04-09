#include "FerroE.H"
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

    amrex::IntVect jx_stag = warpx.get_pointer_current_fp(lev,0)->ixType().toIntVect();
    amrex::IntVect jy_stag = warpx.get_pointer_current_fp(lev,1)->ixType().toIntVect();
    amrex::IntVect jz_stag = warpx.get_pointer_current_fp(lev,2)->ixType().toIntVect();

    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        jx_IndexType[idim]      = jx_stag[idim];
        jy_IndexType[idim]      = jy_stag[idim];
        jz_IndexType[idim]      = jz_stag[idim];
    }

}

void
FerroE::EvolveFerroEJ (amrex::Real dt)
{
    amrex::Print() << " evolve FerroE J using E\n";
    auto & warpx = WarpX::GetInstance();
    const int lev = 0;

    amrex::MultiFab * jx = warpx.get_pointer_current_fp(lev, 0);
    amrex::MultiFab * jy = warpx.get_pointer_current_fp(lev, 1);
    amrex::MultiFab * jz = warpx.get_pointer_current_fp(lev, 2);

    //Px, Py, and Pz have 2 components each. Px(i,j,k,0) is Px and Px(i,j,k,1) is dPx/dt and so on..
    amrex::MultiFab * Px = warpx.get_pointer_polarization_fp(lev, 0);
    amrex::MultiFab * Py = warpx.get_pointer_polarization_fp(lev, 1);
    amrex::MultiFab * Pz = warpx.get_pointer_polarization_fp(lev, 2);


    const amrex::Real mu = 1.35e-18;
    const amrex::Real gamma = 2.0e-7;
    EvolveP(dt, mu, gamma);

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
 *dP/dt = v
 *dv/dt = (E - gamma*v)/mu
 */


// Define the function representing the system of first-order ODEs
AMREX_GPU_DEVICE
void dPdt(amrex::Real P[2], amrex::Real dPdt_result[2], const amrex::Real mu, const amrex::Real gamma, amrex::Real E_eff)
{
     dPdt_result[0] = P[1];
     dPdt_result[1] = E_eff / mu - gamma * P[1] / mu;
}

// Forward Euler time integrator
AMREX_GPU_DEVICE
void forwardEuler(amrex::Real P[2], const amrex::Real dt, const amrex::Real mu, const amrex::Real gamma, amrex::Real E_eff)
{

     amrex::Real res_tmp[2]; // Temporary array to hold the result
     dPdt(P, res_tmp, mu, gamma, E_eff); // Call dPdt with the temporary arrays
     for (int n = 0; n < 2; ++n) P[n] += dt * res_tmp[n]; ;
}

//Evolve P : Integrate equation of motion using Forward Euler method (To Do : Upgrade to higher order integration schemes)
void
FerroE::EvolveP (amrex::Real dt, const amrex::Real mu, const amrex::Real gamma)
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
                   amrex::Real Ex_Landau = 0.0;
                   compute_ex_Landau(Ex_Landau, Px_arr(i,j,k,0), Py_arr(i,j,k,0), Pz_arr(i,j,k,0));
                   Ex_eff += Ex_Landau;
                }

                amrex::Real Px_tmp[2] = {Px_arr(i,j,k,0), Px_arr(i,j,k,1)};
                forwardEuler(Px_tmp, dt, mu, gamma, Ex_eff);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i,j+1,k)==1) {

                amrex::Real Ey_eff = Ey_arr(i,j,k);
                if (include_Landau == 1){
                   amrex::Real Ey_Landau = 0.0;
                   compute_ey_Landau(Ey_Landau, Px_arr(i,j,k,0), Py_arr(i,j,k,0), Pz_arr(i,j,k,0));
                   Ey_eff += Ey_Landau;
                }

	        amrex::Real Py_tmp[2] = {Py_arr(i,j,k,0), Py_arr(i,j,k,1)};
                forwardEuler(Py_tmp, dt, mu, gamma, Ey_eff);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            if (fe_arr(i,j,k)==1 and fe_arr(i,j,k+1)==1) {

                amrex::Real Ez_eff = Ez_arr(i,j,k);
                if (include_Landau == 1){
                   amrex::Real Ez_Landau = 0.0;
                   compute_ez_Landau(Ez_Landau, Px_arr(i,j,k,0), Py_arr(i,j,k,0), Pz_arr(i,j,k,0));
                   Ez_eff += Ez_Landau;
                }

	        amrex::Real Pz_tmp[2] = {Pz_arr(i,j,k,0), Pz_arr(i,j,k,1)};
                forwardEuler(Pz_tmp, dt, mu, gamma, Ez_eff);
            }
        }
    );
    }
}

//Compute x-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_DEVICE
void FerroE::compute_ex_Landau(amrex::Real Ex_Landau, const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    Ex_Landau = alpha_1*Px + alpha_11*std::pow(Px,3.) + alpha_111*std::pow(Px,5.)
               + 2. * alpha_12 * Px * std::pow(Py,2.)
               + 2. * alpha_12 * Px * std::pow(Pz,2.)
               + 4. * alpha_112 * std::pow(Px,3.) * (std::pow(Py,2.) + std::pow(Pz,2.))
               + 2. * alpha_112 * Px * std::pow(Py,4.)
               + 2. * alpha_112 * Px * std::pow(Pz,4.)
               + 2. * alpha_123 * Px * std::pow(Py,2.) * std::pow(Pz,2.);
}


//Compute y-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_DEVICE
void FerroE::compute_ey_Landau(amrex::Real Ey_Landau, const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    Ey_Landau = alpha_1*Py + alpha_11*std::pow(Py,3.) + alpha_111*std::pow(Py,5.)
               + 2. * alpha_12 * Py * std::pow(Px,2.)
               + 2. * alpha_12 * Py * std::pow(Pz,2.)
               + 4. * alpha_112 * std::pow(Py,3.) * (std::pow(Px,2.) + std::pow(Pz,2.))
               + 2. * alpha_112 * Py * std::pow(Px,4.)
               + 2. * alpha_112 * Py * std::pow(Pz,4.)
               + 2. * alpha_123 * Py * std::pow(Px,2.) * std::pow(Pz,2.);
}


//Compute z-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_DEVICE
void FerroE::compute_ez_Landau(amrex::Real Ez_Landau, const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    Ez_Landau = alpha_1*Pz + alpha_11*std::pow(Pz,3.) + alpha_111*std::pow(Pz,5.)
               + 2. * alpha_12 * Pz * std::pow(Px,2.)
               + 2. * alpha_12 * Pz * std::pow(Py,2.)
               + 4. * alpha_112 * std::pow(Pz,3.) * (std::pow(Px,2.) + std::pow(Py,2.))
               + 2. * alpha_112 * Pz * std::pow(Px,4.)
               + 2. * alpha_112 * Pz * std::pow(Py,4.)
               + 2. * alpha_123 * Pz * std::pow(Px,2.) * std::pow(Py,2.);
}
